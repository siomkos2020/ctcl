import os
import random
import sys

import dill

sys.path.append('.')
sys.path.append('..')

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from dataloder.dataset import DatasetForCTMR
from dataloder.utils import pad_train_for_ctmr, pad_num_replace, cal_mortality_metric
from torch.utils.data import DataLoader
from model.multi_modal_encoder import MultiModalForClassification, MultiModalForRegression
import numpy as np
from tqdm import tqdm
import time
import argparse
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--Test', action='store_true', default=False, help='test mode')
parser.add_argument('--task', type=str, default='mortality', help='task type')
parser.add_argument('--resume_path', type=str, default='', help='resume path')
parser.add_argument('--max_text_length', type=int, default=256, help='text length')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--cross_mode', type=str, default='concat', help='')
parser.add_argument('--device', type=int, default=2)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--modality', type=str, default='full')
parser.add_argument('--d_model', type=int, default=256)
parser.add_argument('--pretrained', action='store_true', default=False)
parser.add_argument('--experiment', type=str, default='')

args = parser.parse_args()

set_random_seed(2023)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_name = 'MultiModalForRegression' + args.task + '_' + args.cross_mode + '_' + args.modality + '_' + args.experiment
if not os.path.exists('./saved'):
    os.mkdir('./saved')

if not os.path.exists('./logs'):
    os.mkdir('./logs')

model_saved_path = os.path.join('./saved', model_name)
if not os.path.exists(model_saved_path):
    os.mkdir(model_saved_path)

log_saved_path = os.path.join('./logs', model_name)
if not args.Test and os.path.exists(log_saved_path):
    os.system("rm -r %s"%log_saved_path)

if not os.path.exists(log_saved_path):
    os.mkdir(log_saved_path)

def random_delete_medical_codes(test_data, ratio=0.):
    new_test_data = []
    for feat_set, multiview_graph, text_note, label in test_data:
        feat_set_len = len(feat_set)
        random_del_nums = int(feat_set_len * ratio)
        if random_del_nums == feat_set_len:
            # feat_set = []
            # multiview_graph = {'DIAG-DIAG': [], 'MED-MED': [], 'DIAG-MED': []}
            new_test_data.append((feat_set, multiview_graph, text_note, label))
        else:
            random_ids = np.random.randint(feat_set_len, size=random_del_nums).tolist()
            feat_set = [feat_set[i] for i in range(feat_set_len) if i not in random_ids]
            new_multi_graph = {'DIAG-DIAG': [], 'MED-MED': [], 'DIAG-MED': []}
            for k, edge_set in multiview_graph.items():
                new_edge_set = []
                for edge in edge_set:
                    if edge[0] in feat_set and edge[1] in feat_set:
                        new_edge_set.append(edge)
                new_multi_graph[k] = new_edge_set
            new_test_data.append((feat_set, new_multi_graph, text_note, label))

    return new_test_data


def eval(model, eval_data, eval_batch_size, PAD_TOKEN, device, criterion, epoch, tokenizer):
    model.eval()
    total_batch = len(eval_data) // eval_batch_size
    pred_prob = []
    pred_label = []
    labels = []
    loss_records = []
    mae_records = []
    for i in tqdm(range(total_batch), total=total_batch+1):
        data = pad_train_for_ctmr(eval_data[i*eval_batch_size:(i+1)*eval_batch_size])
        batch_feat_ids, batch_feat_mask, batch_diag_graph, batch_med_graph, batch_dm_graph, batch_labels, batch_text \
            = data
        batch_feat_ids = pad_num_replace(batch_feat_ids, -1, PAD_TOKEN).long().to(device)
        batch_feat_mask = batch_feat_mask.float().to(device)
        batch_diag_graph = batch_diag_graph.float().to(device)
        batch_med_graph = batch_med_graph.float().to(device)
        batch_dm_graph = batch_dm_graph.float().to(device)
        batch_labels = batch_labels.float().to(device)
        if args.modality in ['full', 'text']:
            batch_text = tokenizer(batch_text, padding=True, truncation=True, max_length=args.max_text_length,
                                   return_tensors='pt')

        output, _, _, _ = model(batch_feat_ids, batch_feat_mask, batch_diag_graph, batch_med_graph, batch_dm_graph, batch_text)
        loss_ = criterion(output.squeeze(), batch_labels)
        loss_records.append(loss_.item())
        mae_records.extend(torch.abs(output.squeeze()-batch_labels).detach().cpu().numpy().tolist())
        # exit()

        output = output.squeeze()
        y_pred = output.detach().cpu().numpy()
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0

        pred_prob.extend(output.detach().cpu().numpy().tolist())
        pred_label.extend(y_pred.tolist())
        labels.extend(batch_labels.long().cpu().numpy().tolist())

    if args.task in ['mortality', 'readmission']:
        au_roc, pr_auc, f1, prc, rec = cal_mortality_metric(np.array(pred_prob), np.array(pred_label), np.array(labels))
        print('EPOCH %d, AUROC: %.4f, AUPRC: %.4f, F1: %.4f, PRC: %.4f, REC: %.4f' % (epoch, au_roc, pr_auc, f1, prc, rec))
        model.train()
        return au_roc, pr_auc, f1, np.mean(loss_records)
    else:
        print('EPOCH %d, MSE: %.4f, MAE: %.4f' % (epoch, np.mean(loss_records).item(), np.mean(mae_records).item()))
        model.train()
        return np.mean(mae_records), np.mean(loss_records)

def test(model, test_data, eval_batch_size, PAD_TOKEN, device, tokenizer):
    model.eval()
    total_batch = len(test_data) // eval_batch_size
    pred_prob = []
    pred_label = []
    labels = []
    for i in tqdm(range(total_batch), total=total_batch+1):
        data = pad_train_for_ctmr(test_data[i*eval_batch_size:(i+1)*eval_batch_size])
        batch_feat_ids, batch_feat_mask, batch_diag_graph, batch_med_graph, batch_dm_graph, batch_labels, batch_text \
            = data
        batch_feat_ids = pad_num_replace(batch_feat_ids, -1, PAD_TOKEN).long().to(device)
        batch_feat_mask = batch_feat_mask.float().to(device)
        batch_diag_graph = batch_diag_graph.float().to(device)
        batch_med_graph = batch_med_graph.float().to(device)
        batch_dm_graph = batch_dm_graph.float().to(device)
        batch_labels = batch_labels.float().to(device)
        if args.modality in ['full', 'text']:
            batch_text = tokenizer(batch_text, padding=True, truncation=True, max_length=args.max_text_length,
                                   return_tensors='pt')
        output, _, _, _ = model(batch_feat_ids, batch_feat_mask, batch_diag_graph, batch_med_graph, batch_dm_graph, batch_text)
        y_pred = output.squeeze().detach().cpu().numpy()
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0

        pred_prob.extend(output.squeeze().detach().cpu().numpy().tolist())
        pred_label.extend(y_pred.tolist())
        labels.extend(batch_labels.long().cpu().numpy().tolist())

    return np.array(pred_prob), np.array(pred_label), np.array(labels)

def main():
    EPOCH = 22
    LR = args.lr
    BATCH_SIZE = args.batch_size
    dataset = DatasetForCTMR('../data/eicu/eicu_data.json', task_type=args.task)
    tokenizer = None
    if args.modality in ['full', 'text']:
        tokenizer = AutoTokenizer.from_pretrained("../emilyalsentzer/Bio_ClinicalBERT")
    train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=pad_train_for_ctmr,
                                  shuffle=True, pin_memory=True, num_workers=4)
    feature_voc = dataset.feature2id
    PAD_TOKEN = feature_voc.get('PAD')
    device = torch.device('cuda:%d' % args.device)
    if args.task in ['mortality', 'readmission']:
        model = MultiModalForClassification(input_dim=len(feature_voc), d_model=args.d_model, device=device, num_label=1,
                                            cross_mode=args.cross_mode, modality=args.modality)
    else:
        model = MultiModalForRegression(input_dim=len(feature_voc), d_model=args.d_model, device=device, num_label=1,
                                            cross_mode=args.cross_mode, modality=args.modality)

    if args.pretrained:
        tmp_model = MultiModalForClassification(input_dim=len(feature_voc), d_model=args.d_model, device=device,
                                            num_label=1,
                                            cross_mode=args.cross_mode, modality=args.modality)
        print('Load pretrained weight...')
        tmp_model.load_state_dict(torch.load(args.resume_path))
        model.from_pretrained(tmp_model.multi_model_enc.state_dict())
        del tmp_model


    model.to(device)

    if args.Test:
        model.load_state_dict(torch.load(open(args.resume_path, 'rb'), map_location=device))
        test_data = random_delete_medical_codes(dataset.test_data, ratio=0.9)
        y_prob, y_pred, y_label = test(model, test_data, BATCH_SIZE, PAD_TOKEN, device, tokenizer)
        result = []
        for _ in range(10):
            data_num = len(y_prob)
            final_length = int(0.8 * data_num)
            idx_list = list(range(data_num))
            random.shuffle(idx_list)
            idx_list = idx_list[:final_length]
            au_roc, pr_auc, f1, _, _ = cal_mortality_metric(y_prob[idx_list], y_pred[idx_list], y_label[idx_list])
            result.append([au_roc, pr_auc, f1])


        result = np.array(result)
        mean = result.mean(axis=0)
        std = result.std(axis=0)

        outstring = ""
        for m, s in zip(mean, std):
            outstring += "{:.4f} $\pm$ {:.4f} & ".format(m, s)
        print(outstring)
        exit()


    writter = SummaryWriter(log_dir=log_saved_path)
    if args.task in ['mortality', 'readmission']:
        criterion = nn.BCELoss()
    else:
        criterion = nn.MSELoss()

    loss_records = {
        'train-loss': [],
        'test-loss':  [],
        'train-time': [],
        'mse_loss': [],
        'mae_metric': []
    }
    # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(dataset.pos_weight).to(device))
    if args.modality == 'graph':
        optimizer = Adam(model.parameters(), lr=LR)
    elif args.modality == 'text':
        optimizer = AdamW(model.parameters(), lr=LR)
    else:
        optimizer = AdamW(model.parameters(), lr=LR)
        # optimizer1 = AdamW(model.multi_model_enc.parameters(), lr=1e-5)
        # optimizer2 = AdamW(model.output_layer.parameters(), lr=3e-4)

    # if args.task == 'LOS':
    #     optimizer = SGD(model.parameters(), lr=LR)

    global_step = 0
    best_performance = (1000000, 0, 0)
    epoch_steps = len(dataset) / BATCH_SIZE
    model.train()
    for epoch in range(EPOCH):
        epoch_loss = []
        start_epoch = time.time()
        for idx, data in enumerate(train_dataloader):
            batch_feat_ids, batch_feat_mask, batch_diag_graph, batch_med_graph, batch_dm_graph, batch_labels, batch_text\
                = data
            batch_feat_ids = pad_num_replace(batch_feat_ids, -1, PAD_TOKEN).long().to(device)
            batch_feat_mask = batch_feat_mask.float().to(device)
            batch_diag_graph = batch_diag_graph.float().to(device)
            batch_med_graph = batch_med_graph.float().to(device)
            batch_dm_graph = batch_dm_graph.float().to(device)
            batch_labels = batch_labels.float().to(device)
            # print(batch_text[0])
            # if args.modality in ['full', 'text']:
            #     batch_text = tokenizer(batch_text, padding=True, truncation=True, max_length=args.max_text_length, return_tensors='pt')

            # print(tokenizer.convert_ids_to_tokens(batch_text['input_ids'].numpy().tolist()[0]))
            # exit()
            output, cont_loss,_,_ = model(batch_feat_ids, batch_feat_mask, batch_diag_graph, batch_med_graph, batch_dm_graph, batch_text,
                                      tokenizer)
            # output, _ = model(batch_feat_ids, batch_feat_mask, batch_diag_graph, batch_med_graph,
            #                           batch_dm_graph, batch_text)
            # cont_loss = torch.tensor(0.)
            loss = criterion(output.squeeze(), batch_labels) #+ 0.1*cont_loss
            loss_records['train-loss'].append(loss.item())
            epoch_loss.append(loss.item())
            # optimizer1.zero_grad()
            # optimizer2.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # optimizer1.step()
            # optimizer2.step()
            writter.add_scalar('loss/train', loss.item(), global_step=global_step)
            if global_step % 50 == 0:
                print('EPOCH %d Step: %d/%d loss: %.4f, Cont loss: %.4f'%(epoch, idx, epoch_steps, loss.item(), cont_loss.item()))
            global_step += 1

        if args.modality != 'text':
            end_time = time.time()
            if args.task in ['mortality', 'readmission']:
                au_roc, pr_auc, f1, mean_loss = eval(model, dataset.test_data, BATCH_SIZE, PAD_TOKEN, device, criterion, epoch, tokenizer)
                loss_records['test-loss'].append(mean_loss)
                loss_records['train-time'].append(end_time-start_epoch)
                writter.add_scalars('metrics', {'AUROC': au_roc, 'AUPRC': pr_auc, 'F1': f1}, global_step=global_step)
                if au_roc > best_performance[0] and \
                    pr_auc > best_performance[1]:
                    best_performance = (au_roc, pr_auc, f1)
                    model_weight_path = os.path.join(model_saved_path, '%s_epoch_%d_au_%.4f.pt'%(args.d_model, epoch, au_roc))
                    torch.save(model.state_dict(), model_weight_path)
                    print('model saved..')
                writter.add_scalars('loss/avg_loss', {'eval': mean_loss.item(),
                                                      'train': np.mean(epoch_loss).item()}, global_step=global_step)
                print('EPOCH %d avg loss: %.4f, Time Cost: %.4f s'%(epoch, np.mean(epoch_loss).item(),
                                                                  end_time-start_epoch))
            else:
                mae_metric, mse_metric = eval(model, dataset.test_data, BATCH_SIZE, PAD_TOKEN, device, criterion,
                                                     epoch, tokenizer)
                loss_records['test-loss'].append(mse_metric)
                loss_records['train-time'].append(end_time - start_epoch)
                loss_records['mse_loss'].append(mse_metric)
                loss_records['mae_metric'].append(mae_metric)
                if mse_metric < best_performance[0]:
                    best_performance = (mse_metric, 0, 0)
                    model_weight_path = os.path.join(model_saved_path,
                                                     '%s_epoch_%d_au_%.4f.pt' % (args.d_model, epoch, mse_metric))
                    torch.save(model.state_dict(), model_weight_path)
                    print('model saved..')
                print('EPOCH %d avg loss: %.4f, Time Cost: %.4f s' % (epoch, np.mean(epoch_loss).item(),
                                                                      end_time - start_epoch))

    dill.dump(loss_records, open(os.path.join(model_saved_path, '%s_loss_records'%args.d_model),
                                 'wb'))




if __name__ == '__main__':
    main()