import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


def pad_train_for_graph(batch):
    batch_size = len(batch)
    feat_size = torch.tensor([len(adm[0]) for adm in batch])
    max_feat_size = max(feat_size)

    batch_feat_ids = torch.full((batch_size, max_feat_size), -1)
    batch_feat_mask = torch.zeros((batch_size, max_feat_size))
    batch_adjacency = torch.zeros((batch_size, max_feat_size, max_feat_size))
    batch_labels = torch.zeros((batch_size))
    for batch_i, (feat, graph, _, label) in enumerate(batch):
        feat_len = len(feat)
        featid2adjid = {k:i for i, k in enumerate(feat)}
        batch_feat_ids[batch_i, :feat_len] = torch.tensor(feat)
        batch_feat_mask[batch_i, :feat_len] = 1
        batch_labels[batch_i] = label
        for edge in graph:
            if len(edge) == 2:
                batch_adjacency[batch_i, featid2adjid[edge[0]], featid2adjid[edge[1]]] = 1
                batch_adjacency[batch_i, featid2adjid[edge[1]], featid2adjid[edge[0]]] = 1
            else:
                batch_adjacency[batch_i, featid2adjid[edge[0]], featid2adjid[edge[1]]] = edge[2]
                # batch_adjacency[batch_i, featid2adjid[edge[1]], featid2adjid[edge[0]]] = edge[2]

        for feat_i in featid2adjid.values():
            batch_adjacency[batch_i, feat_i, feat_i] = 1


    return batch_feat_ids, batch_feat_mask, batch_adjacency, batch_labels


def pad_train_for_gct(batch):
    batch_size = len(batch)
    feat_size = torch.tensor([len(adm[0]) for adm in batch])
    max_feat_size = max(feat_size)

    batch_feat_ids = torch.full((batch_size, max_feat_size), -1)
    batch_feat_mask = torch.full((batch_size, max_feat_size, max_feat_size), -1e9)
    batch_cond_p = torch.zeros((batch_size, max_feat_size, max_feat_size))

    batch_labels = torch.zeros((batch_size))
    for batch_i, (feat, feat_mask, cond_p, label) in enumerate(batch):
        feat_len = len(feat)
        featid2adjid = {k:i for i, k in enumerate(feat)}
        batch_feat_ids[batch_i, :feat_len] = torch.tensor(feat)
        batch_labels[batch_i] = label
        for edge in feat_mask:
            batch_feat_mask[batch_i, featid2adjid[edge[0]], featid2adjid[edge[1]]] = 0
                # batch_adjacency[batch_i, featid2adjid[edge[1]], featid2adjid[edge[0]]] = edge[2]

        for feat_i in featid2adjid.values():
            batch_feat_mask[batch_i, feat_i, feat_i] = 0

        for pair in cond_p:
            batch_cond_p[batch_i, featid2adjid[pair[0]], featid2adjid[pair[1]]] = pair[2]

    batch_cond_p_sum = batch_cond_p.sum(-1).unsqueeze(-1)
    batch_cond_p_sum[batch_cond_p_sum == 0] = 1e-4
    batch_cond_p = batch_cond_p / batch_cond_p_sum

    return batch_feat_ids, batch_feat_mask, batch_cond_p, batch_labels


def pad_train_for_multigraph(batch):
    batch_size = len(batch)
    feat_size = torch.tensor([len(adm[0]) for adm in batch])
    max_feat_size = max(feat_size)

    batch_feat_ids = torch.full((batch_size, max_feat_size), -1)
    batch_feat_mask = torch.zeros((batch_size, max_feat_size))
    batch_diag_graph = torch.zeros((batch_size, max_feat_size, max_feat_size))
    batch_med_graph = torch.zeros((batch_size, max_feat_size, max_feat_size))
    batch_dm_graph = torch.zeros((batch_size, max_feat_size, max_feat_size))

    batch_labels = torch.zeros((batch_size))
    for batch_i, (feat, multi_graph, label) in enumerate(batch):
        feat_len = len(feat)
        featid2adjid = {k:i for i, k in enumerate(feat)}
        batch_feat_ids[batch_i, :feat_len] = torch.tensor(feat)
        batch_feat_mask[batch_i, :feat_len] = 1
        batch_labels[batch_i] = label

        for edge in multi_graph['DIAG-DIAG']:
            batch_diag_graph[batch_i, featid2adjid[edge[0]], featid2adjid[edge[1]]] = 1
        for edge in multi_graph['MED-MED']:
            batch_med_graph[batch_i, featid2adjid[edge[0]], featid2adjid[edge[1]]] = 1
        for edge in multi_graph['DIAG-MED']:
            batch_dm_graph[batch_i, featid2adjid[edge[0]], featid2adjid[edge[1]]] = 1

        for feat_i in featid2adjid.values():
            batch_diag_graph[batch_i, feat_i, feat_i] = 1
            batch_med_graph[batch_i, feat_i, feat_i] = 1
            batch_dm_graph[batch_i, feat_i, feat_i] = 1


    return batch_feat_ids, batch_feat_mask, batch_diag_graph, batch_med_graph, batch_dm_graph, batch_labels


def pad_train_for_ctmr(batch):
    batch_size = len(batch)
    feat_size = torch.tensor([len(adm[0]) for adm in batch])
    max_feat_size = max(feat_size)

    batch_feat_ids = torch.full((batch_size, max_feat_size), -1)
    batch_feat_mask = torch.full((batch_size, max_feat_size), -1e9)
    batch_diag_graph = torch.zeros((batch_size, max_feat_size, max_feat_size))
    batch_med_graph = torch.zeros((batch_size, max_feat_size, max_feat_size))
    batch_dm_graph = torch.zeros((batch_size, max_feat_size, max_feat_size))
    batch_text = []

    batch_labels = torch.zeros((batch_size))
    for batch_i, (feat, multi_graph, note_text, label) in enumerate(batch):
        note_text = note_text.lower()
        note_text = note_text.replace('diagnoses:', '').replace('medications:', '')
        note_text = note_text.replace('|', ' ')
        note_text = note_text.replace('/', ' ')
        batch_text.append(note_text)
        feat_len = len(feat)
        featid2adjid = {k:i for i, k in enumerate(feat)}
        batch_feat_ids[batch_i, :feat_len] = torch.tensor(feat)
        batch_feat_mask[batch_i, :feat_len] = 0
        batch_labels[batch_i] = label

        for edge in multi_graph['DIAG-DIAG']:
            batch_diag_graph[batch_i, featid2adjid[edge[0]], featid2adjid[edge[1]]] = 1
        for edge in multi_graph['MED-MED']:
            batch_med_graph[batch_i, featid2adjid[edge[0]], featid2adjid[edge[1]]] = 1
        for edge in multi_graph['DIAG-MED']:
            batch_dm_graph[batch_i, featid2adjid[edge[0]], featid2adjid[edge[1]]] = 1

        for feat_i in featid2adjid.values():
            batch_diag_graph[batch_i, feat_i, feat_i] = 1
            batch_med_graph[batch_i, feat_i, feat_i] = 1
            batch_dm_graph[batch_i, feat_i, feat_i] = 1


    return batch_feat_ids, batch_feat_mask, batch_diag_graph, batch_med_graph, batch_dm_graph, batch_labels, batch_text

def pad_pretrain_for_ctmr(batch):
    # print(batch)
    batch_size = len(batch)
    feat_size = torch.tensor([len(adm[0]) for adm in batch])
    max_feat_size = max(feat_size)

    batch_feat_ids = torch.full((batch_size, max_feat_size), -1)
    batch_feat_mask = torch.full((batch_size, max_feat_size), -1e9)
    batch_diag_graph = torch.zeros((batch_size, max_feat_size, max_feat_size))
    batch_med_graph = torch.zeros((batch_size, max_feat_size, max_feat_size))
    batch_dm_graph = torch.zeros((batch_size, max_feat_size, max_feat_size))
    batch_text = []
    batch_mask_pos = []

    batch_labels = torch.zeros((batch_size))
    for batch_i, (raw_feat, feat, multi_graph, note_text, label, MASK_TOKEN, mask_pos) in enumerate(batch):
        # print(raw_feat)
        # print(feat)
        # print(multi_graph)
        # print(note_text)
        # print(label)
        # print(MASK_TOKEN)
        # print(mask_pos)
        batch_mask_pos.append(mask_pos)
        note_text = note_text.lower()
        note_text = note_text.replace('diagnoses:', '').replace('medications:', '')
        note_text = note_text.replace('|', ' ')
        note_text = note_text.replace('/', ' ')
        batch_text.append(note_text)
        feat_len = len(feat)
        featid2adjid = {k:i for i, k in enumerate(raw_feat)}
        batch_feat_ids[batch_i, :feat_len] = torch.tensor(feat)
        batch_feat_mask[batch_i, :feat_len] = 0
        # batch_feat_mask[batch_i, mask_pos] = -1e9
        batch_labels[batch_i] = label

        for edge in multi_graph['DIAG-DIAG']:
            # if label not in edge:
            batch_diag_graph[batch_i, featid2adjid[edge[0]], featid2adjid[edge[1]]] = 1
        for edge in multi_graph['MED-MED']:
            # if label not in edge:
            batch_med_graph[batch_i, featid2adjid[edge[0]], featid2adjid[edge[1]]] = 1
        for edge in multi_graph['DIAG-MED']:
            # if label not in edge:
            batch_dm_graph[batch_i, featid2adjid[edge[0]], featid2adjid[edge[1]]] = 1

        for feat_i in featid2adjid.values():
            batch_diag_graph[batch_i, feat_i, feat_i] = 1
            batch_med_graph[batch_i, feat_i, feat_i] = 1
            batch_dm_graph[batch_i, feat_i, feat_i] = 1


    return batch_feat_ids, batch_feat_mask, batch_diag_graph, batch_med_graph, \
           batch_dm_graph, batch_labels, batch_text, torch.LongTensor(batch_mask_pos)

def pad_pretrain_for_ctmr_v1(batch):
    # print(batch)
    batch_size = len(batch)
    batch_return = {
        'mask_code': [],
        'med_pred_diag': []
    }
    # batch_mask_code = batch.get('mask_code')
    # batch_med_pred_diag = batch.get('med_pred_diag')
    feat_size = torch.tensor([len(adm.get('mask_code')[0]) for adm in batch])
    max_feat_size = max(feat_size)

    batch_feat_ids = torch.full((batch_size, max_feat_size), -1)
    batch_feat_mask = torch.full((batch_size, max_feat_size), -1e9)
    batch_diag_graph = torch.zeros((batch_size, max_feat_size, max_feat_size))
    batch_med_graph = torch.zeros((batch_size, max_feat_size, max_feat_size))
    batch_dm_graph = torch.zeros((batch_size, max_feat_size, max_feat_size))
    batch_text = []
    batch_mask_pos = []
    label_size = batch[0].get('feat_voc_size')
    batch_labels = torch.zeros((batch_size, label_size))
    for batch_i in range(batch_size):
        raw_feat, feat, multi_graph, note_text, label, MASK_TOKEN, mask_pos = batch[batch_i].get('mask_code')
        # print(raw_feat)
        # print(feat)
        # print(multi_graph)
        # print(note_text)
        # print(label)
        # print(MASK_TOKEN)
        # print(mask_pos)
        batch_mask_pos.append(mask_pos)
        note_text = note_text.lower()
        note_text = note_text.replace('diagnoses:', '').replace('medications:', '')
        note_text = note_text.replace('|', ' ')
        note_text = note_text.replace('/', ' ')
        batch_text.append(note_text)
        feat_len = len(feat)
        featid2adjid = {k:i for i, k in enumerate(raw_feat)}
        batch_feat_ids[batch_i, :feat_len] = torch.tensor(raw_feat)
        batch_feat_mask[batch_i, :feat_len] = 0
        # batch_feat_mask[batch_i, mask_pos] = -1e9
        batch_labels[batch_i, raw_feat] = 1

        for edge in multi_graph['DIAG-DIAG']:
            # if label not in edge:
            batch_diag_graph[batch_i, featid2adjid[edge[0]], featid2adjid[edge[1]]] = 1
        for edge in multi_graph['MED-MED']:
            # if label not in edge:
            batch_med_graph[batch_i, featid2adjid[edge[0]], featid2adjid[edge[1]]] = 1
        for edge in multi_graph['DIAG-MED']:
            # if label not in edge:
            batch_dm_graph[batch_i, featid2adjid[edge[0]], featid2adjid[edge[1]]] = 1

        for feat_i in featid2adjid.values():
            batch_diag_graph[batch_i, feat_i, feat_i] = 1
            batch_med_graph[batch_i, feat_i, feat_i] = 1
            batch_dm_graph[batch_i, feat_i, feat_i] = 1

    batch_return['mask_code'] = (
        batch_feat_ids, batch_feat_mask, batch_diag_graph, batch_med_graph,
        batch_dm_graph, batch_labels, batch_text, torch.LongTensor(batch_mask_pos)
    )

    ## Task 2
    feat_size = [len(adm.get('med_pred_diag')[0]) for adm in batch]
    max_feat_size = max(feat_size)
    label_size = batch[0].get('diag_num')

    batch_feat_ids = torch.full((batch_size, max_feat_size), -1)
    batch_feat_mask = torch.full((batch_size, max_feat_size), -1e9)
    batch_diag_graph = torch.zeros((batch_size, max_feat_size, max_feat_size))
    batch_med_graph = torch.zeros((batch_size, max_feat_size, max_feat_size))
    batch_dm_graph = torch.zeros((batch_size, max_feat_size, max_feat_size))
    batch_labels = torch.zeros((batch_size, label_size))
    for batch_i in range(batch_size):
        med_feat, multi_view_graph, _, diag_feat = batch[batch_i].get('med_pred_diag')
        batch_feat_ids[batch_i, :len(med_feat)] = torch.tensor(med_feat)
        batch_feat_mask[batch_i, :len(med_feat)] = 0
        featid2adjid = {k: i for i, k in enumerate(med_feat)}
        batch_labels[batch_i, diag_feat] = 1
        for edge in multi_view_graph['DIAG-DIAG']:
            batch_diag_graph[batch_i, featid2adjid[edge[0]], featid2adjid[edge[1]]] = 1
        for edge in multi_view_graph['MED-MED']:
            batch_med_graph[batch_i, featid2adjid[edge[0]], featid2adjid[edge[1]]] = 1
        for edge in multi_view_graph['DIAG-MED']:
            batch_dm_graph[batch_i, featid2adjid[edge[0]], featid2adjid[edge[1]]] = 1

        for feat_i in featid2adjid.values():
            batch_diag_graph[batch_i, feat_i, feat_i] = 1
            batch_med_graph[batch_i, feat_i, feat_i] = 1
            batch_dm_graph[batch_i, feat_i, feat_i] = 1

    batch_return['med_pred_diag'] = (
        batch_feat_ids, batch_feat_mask, batch_diag_graph, batch_med_graph,
        batch_dm_graph, batch_labels, batch_text
    )

    ##Task 3
    feat_size = [len(adm.get('diag_pred_med')[0]) for adm in batch]
    max_feat_size = max(feat_size)
    label_size = batch[0].get('med_num')

    batch_feat_ids = torch.full((batch_size, max_feat_size), -1)
    batch_feat_mask = torch.full((batch_size, max_feat_size), -1e9)
    batch_diag_graph = torch.zeros((batch_size, max_feat_size, max_feat_size))
    batch_med_graph = torch.zeros((batch_size, max_feat_size, max_feat_size))
    batch_dm_graph = torch.zeros((batch_size, max_feat_size, max_feat_size))
    batch_labels = torch.zeros((batch_size, label_size))
    for batch_i in range(batch_size):
        med_feat, multi_view_graph, _, diag_feat = batch[batch_i].get('diag_pred_med')
        batch_feat_ids[batch_i, :len(med_feat)] = torch.tensor(med_feat)
        batch_feat_mask[batch_i, :len(med_feat)] = 0
        featid2adjid = {k: i for i, k in enumerate(med_feat)}
        batch_labels[batch_i, diag_feat] = 1
        for edge in multi_view_graph['DIAG-DIAG']:
            batch_diag_graph[batch_i, featid2adjid[edge[0]], featid2adjid[edge[1]]] = 1
        for edge in multi_view_graph['MED-MED']:
            batch_med_graph[batch_i, featid2adjid[edge[0]], featid2adjid[edge[1]]] = 1
        for edge in multi_view_graph['DIAG-MED']:
            batch_dm_graph[batch_i, featid2adjid[edge[0]], featid2adjid[edge[1]]] = 1

        for feat_i in featid2adjid.values():
            batch_diag_graph[batch_i, feat_i, feat_i] = 1
            batch_med_graph[batch_i, feat_i, feat_i] = 1
            batch_dm_graph[batch_i, feat_i, feat_i] = 1

    batch_return['diag_pred_med'] = (
        batch_feat_ids, batch_feat_mask, batch_diag_graph, batch_med_graph,
        batch_dm_graph, batch_labels, batch_text
    )


    return batch_return


def pad_train_for_hcl(batch):
    batch_size = len(batch)
    feat_size = torch.tensor([len(adm[0]) for adm in batch])
    max_feat_size = max(feat_size)
    # Build code id to temp id
    code_id2tempid = {}
    for adm in batch:
        for c in adm[0]:
            if c not in code_id2tempid:
                code_id2tempid[c] = len(code_id2tempid)

    batch_feat_ids = torch.full((batch_size, max_feat_size), -1)
    batch_feat_mask = torch.zeros((batch_size, max_feat_size))
    batch_code_set = list(code_id2tempid.keys())
    batch_pc_adj = torch.zeros((batch_size, len(batch_code_set)))
    batch_pp_adj = torch.zeros((batch_size, batch_size))
    batch_labels = torch.zeros((batch_size))
    code_tf = torch.zeros((batch_size, len(batch_code_set)))
    code_idf = torch.zeros((batch_size, len(batch_code_set)))

    for batch_i, (feat, graph, _, label) in enumerate(batch):
        feat_len = len(feat)
        batch_feat_ids[batch_i, :feat_len] = torch.tensor(feat)
        batch_feat_mask[batch_i, :feat_len] = 1
        batch_labels[batch_i] = label
        feat_t_ids = [code_id2tempid[c] for c in feat]
        for idx in feat_t_ids:
            batch_pc_adj[batch_i, idx] = 1.
            code_tf[batch_i, idx] = 1
            code_idf[batch_i, idx] = 1

    code_tf = code_tf / (code_tf.sum(-1).unsqueeze(-1) + 1e-6)
    code_idf = batch_size / (code_idf.sum(0) + 1e-6)

    batch_p_feat = code_tf * code_idf.unsqueeze(0)
    batch_sim_mat = torch.mm(batch_p_feat, batch_p_feat.transpose(0, 1))
    for i in range(batch_size):
        idx_dict = {idx:sim for idx, sim in enumerate(batch_sim_mat[i].numpy().tolist())}
        idx_list = sorted(idx_dict.items(), key=lambda x:x[1], reverse=True)[:7]
        idx_list = [idx[0] for idx in idx_list]
        batch_pp_adj[i, idx_list] = 1

    return batch_feat_ids, batch_feat_mask, batch_labels, torch.LongTensor(batch_code_set),  batch_pc_adj, batch_pp_adj


def pad_train_for_hcl_bert(batch):
    batch_size = len(batch)
    feat_size = torch.tensor([len(adm[0]) for adm in batch])
    max_feat_size = max(feat_size)
    # Build code id to temp id
    code_id2tempid = {}
    for adm in batch:
        for c in adm[0]:
            if c not in code_id2tempid:
                code_id2tempid[c] = len(code_id2tempid)

    batch_feat_ids = torch.full((batch_size, max_feat_size), -1)
    batch_feat_mask = torch.zeros((batch_size, max_feat_size))
    batch_code_set = list(code_id2tempid.keys())
    batch_pc_adj = torch.zeros((batch_size, len(batch_code_set)))
    batch_pp_adj = torch.zeros((batch_size, batch_size))
    batch_labels = torch.zeros((batch_size))
    code_tf = torch.zeros((batch_size, len(batch_code_set)))
    code_idf = torch.zeros((batch_size, len(batch_code_set)))
    batch_text = []

    for batch_i, (feat, graph, note_text, label) in enumerate(batch):
        feat_len = len(feat)
        note_text = note_text.lower()
        note_text = note_text.replace('diagnoses:', '').replace('medications:', '')
        note_text = note_text.replace('|', ' ')
        note_text = note_text.replace('/', ' ')
        batch_text.append(note_text)

        batch_feat_ids[batch_i, :feat_len] = torch.tensor(feat)
        batch_feat_mask[batch_i, :feat_len] = 1
        batch_labels[batch_i] = label
        feat_t_ids = [code_id2tempid[c] for c in feat]
        for idx in feat_t_ids:
            batch_pc_adj[batch_i, idx] = 1.
            code_tf[batch_i, idx] = 1
            code_idf[batch_i, idx] = 1

    code_tf = code_tf / (code_tf.sum(-1).unsqueeze(-1) + 1e-6)
    code_idf = batch_size / (code_idf.sum(0) + 1e-6)

    batch_p_feat = code_tf * code_idf.unsqueeze(0)
    batch_sim_mat = torch.mm(batch_p_feat, batch_p_feat.transpose(0, 1))
    for i in range(batch_size):
        idx_dict = {idx:sim for idx, sim in enumerate(batch_sim_mat[i].numpy().tolist())}
        idx_list = sorted(idx_dict.items(), key=lambda x:x[1], reverse=True)[:7]
        idx_list = [idx[0] for idx in idx_list]
        batch_pp_adj[i, idx_list] = 1

    return batch_feat_ids, batch_feat_mask, batch_labels, torch.LongTensor(batch_code_set),  \
           batch_pc_adj, batch_pp_adj, batch_text





def pad_num_replace(tensor, src_num, target_num):
    return torch.where(tensor==src_num, target_num, tensor)

def cal_mortality_metric(y_prob, y_pred, labels):
    prc = ((y_pred == 1) & (labels == 1)).sum() / (y_pred == 1).sum() if (y_pred == 1).sum() > 0 else 0
    rec = ((y_pred == 1) & (labels == 1)).sum() / (labels == 1).sum()
    au_roc = roc_auc_score(labels, y_prob)
    pr_auc = average_precision_score(labels, y_prob, average='macro')
    f1 = f1_score(y_pred, labels)

    return au_roc, pr_auc, f1, prc, rec
