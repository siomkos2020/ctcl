import os
import torch
import json
import dill
from torch.utils.data import Dataset
from prettytable import PrettyTable
import numpy as np
from tqdm import tqdm

class DatasetForGraph(Dataset):
    def __init__(self, raw_data_path, task_type,  graph_mode=None):
        super(DatasetForGraph, self).__init__()
        self.root_dir = '/'.join(raw_data_path.split('/')[:-1])
        self.graph_mode = graph_mode
        self.task_type = task_type
        self.feature2id = json.load(open(os.path.join(self.root_dir, 'feature2id.json'), 'r', encoding='utf-8'))
        self.id2feature = {v: k for k, v in self.feature2id.items()}
        self._graph_mode_construct()
        self.load_data()

    def _graph_mode_construct(self):
        if self.graph_mode == 'random':
            if os.path.exists(os.path.join(self.root_dir, 'random_graph.pkl')):
                self.graph = dill.load(open(os.path.join(self.root_dir, 'random_graph.pkl'), 'rb'))
        elif self.graph_mode == 'conditional':
            if os.path.exists(os.path.join(self.root_dir, 'conditional_graph.pkl')):
                self.graph = dill.load(open(os.path.join(self.root_dir, 'conditional_graph.pkl'), 'rb'))

    def reset_graph(self, data):
        new_data = []
        for feat_set, text_feat, label in data:
            feat_set = [self.feature2id[x] for x in feat_set]
            cond_graph = []
            for i in range(len(feat_set)):
                for j in range(i+1, len(feat_set)):
                    edge = (feat_set[i], feat_set[j])
                    if self.graph.get(edge, None) is not None:
                        cond_graph.append((feat_set[i], feat_set[j], self.graph.get(edge)))
            new_data.append((feat_set, cond_graph, text_feat, label))

        return new_data

    def load_data(self):
        if os.path.exists(os.path.join(self.root_dir, 'downstream', self.task_type, 'train.pkl')):
            self.train_data = dill.load(open(os.path.join(self.root_dir, 'downstream', self.task_type, 'train.pkl'), 'rb'))
            self.eval_data = dill.load(open(os.path.join(self.root_dir, 'downstream', self.task_type, 'eval.pkl'), 'rb'))
            self.test_data = dill.load(open(os.path.join(self.root_dir, 'downstream', self.task_type, 'test.pkl'), 'rb'))

            if self.graph_mode in ['random', 'conditional']:
                self.train_data = self.reset_graph(self.train_data)
                self.eval_data = self.reset_graph(self.eval_data)
                self.test_data = self.reset_graph(self.test_data)

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item):
        return self.train_data[item]


class DatasetForGCT(Dataset):
    def __init__(self, raw_data_path, task_type):
        super(DatasetForGCT, self).__init__()
        self.root_dir = '/'.join(raw_data_path.split('/')[:-1])
        self.task_type = task_type
        self.build_condition_graph()
        self._load_data()

    def _load_data(self):
        print("Building medical code vocabulary...")
        self._build_feature_voc()
        self.train_data = dill.load(open(os.path.join(self.root_dir, 'downstream', self.task_type, 'train.pkl'), 'rb'))
        self.eval_data = dill.load(open(os.path.join(self.root_dir, 'downstream', self.task_type, 'eval.pkl'), 'rb'))
        self.test_data = dill.load(open(os.path.join(self.root_dir, 'downstream', self.task_type, 'test.pkl'), 'rb'))

        self.train_data = self.patch_data(self.train_data)
        self.eval_data = self.patch_data(self.eval_data)
        self.test_data = self.patch_data(self.test_data)


    def build_condition_graph(self):
        # Construct conditional probability matrix (ignore correlation between codes of same type)
        self.condition_graph = dill.load(open(os.path.join(self.root_dir, 'conditional_graph.pkl'), 'rb'))

    def _build_feature_voc(self):
        self.feature2id = json.load(open(os.path.join(self.root_dir, 'feature2id.json'), 'r', encoding='utf-8'))
        self.feature2id['[CLS]'] = len(self.feature2id)
        self.id2feature = {v: k for k, v in self.feature2id.items()}

    def __len__(self):
        return len(self.train_data)

    def get_gct_mask(self, feat_set):
        feat_mask = []
        cond_p = []
        for feat_id in feat_set:
            if 'DIAG' in self.id2feature[feat_id]:
                feat_mask.append((self.feature2id['[CLS]'], feat_id))
                feat_mask.append((feat_id, self.feature2id['[CLS]']))

        for i_x, feat_x in enumerate(feat_set):
            for i_y, feat_y in enumerate(feat_set):
                if self.condition_graph.get((feat_x, feat_y), None) is not None:
                    cond_p.append((feat_x, feat_y, self.condition_graph.get((feat_x, feat_y))))
                    feat_mask.append((feat_x, feat_y))

        return feat_mask, cond_p

    def patch_data(self, data):
        new_data = []
        for feat_set, _, label in data:
            feat_set = [self.feature2id[x] for x in feat_set]
            feat_mask, cond_p = self.get_gct_mask(feat_set)
            new_data.append(([self.feature2id['[CLS]']] + feat_set, feat_mask, cond_p, label))
        return new_data

    def __getitem__(self, item):
        return self.train_data[item]


class DatasetForRGraph(Dataset):
    def __init__(self, raw_data_path, task_type):
        super(DatasetForRGraph, self).__init__()
        self.root_dir = '/'.join(raw_data_path.split('/')[:-1])
        self.task_type = task_type
        self._load_data()

    def _load_data(self):
        print("Building medical code vocabulary...")
        self._build_feature_voc()
        self.train_data = dill.load(open(os.path.join(self.root_dir, 'downstream', self.task_type, 'train.pkl'), 'rb'))
        self.eval_data = dill.load(open(os.path.join(self.root_dir, 'downstream', self.task_type, 'eval.pkl'), 'rb'))
        self.test_data = dill.load(open(os.path.join(self.root_dir, 'downstream', self.task_type, 'test.pkl'), 'rb'))
        self._build_corr_graph()
        self.train_data = self.patch_data(self.train_data)
        self.eval_data = self.patch_data(self.eval_data)
        self.test_data = self.patch_data(self.test_data)

    def patch_data(self, data):
        new_data = []
        for feat_set, _,  label in data:
            feat_set = [self.feature2id[x] for x in feat_set]
            multiview_graph = {'DIAG-DIAG': [], 'MED-MED': [], 'DIAG-MED': []}
            for feat_x in feat_set:
                feat_x_type = 'DIAG' if 'DIAG' in self.id2feature[feat_x] else 'MED'
                for feat_y in feat_set:
                    feat_y_type = 'DIAG' if 'DIAG' in self.id2feature[feat_y] else 'MED'

                    if self.cor_graph.get(feat_x).get(feat_y) > 1:
                        if feat_x_type == feat_y_type:
                            multiview_graph[feat_x_type+'-'+feat_y_type].append((feat_x, feat_y))
                        else:
                            multiview_graph['DIAG-MED'].append((feat_x, feat_y))
            new_data.append((feat_set, multiview_graph, label))
        return new_data

    def _build_feature_voc(self):
        self.feature2id = json.load(
            open(os.path.join(self.root_dir, 'feature2id.json'), 'r', encoding='utf-8'))
        self.id2feature = {v: k for k, v in self.feature2id.items()}

    def _build_corr_graph(self):
        if os.path.exists(os.path.join(self.root_dir, '%s_cooccurrence_graph.pkl'%self.task_type)):
            self.cor_graph = dill.load(open(os.path.join(self.root_dir, '%s_cooccurrence_graph.pkl'%self.task_type), 'rb'))
        else:
            data = self.train_data + self.eval_data + self.test_data
            cor_graph = {}
            for feat_set, _, label in data:
                feat_set = [self.feature2id[x] for x in feat_set]
                for feat_x in feat_set:
                    for feat_y in feat_set:
                        if feat_x not in cor_graph:
                            cor_graph[feat_x] = {}
                        if feat_y not in cor_graph[feat_x]:
                            cor_graph[feat_x][feat_y] = 0
                        cor_graph[feat_x][feat_y] += 1
            self.cor_graph = cor_graph
            dill.dump(self.cor_graph, open(os.path.join(self.root_dir, '%s_cooccurrence_graph.pkl'%self.task_type), 'wb'))

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item):
        return self.train_data[item]


class DatasetForCTMR(Dataset):
    def __init__(self, raw_data_path, task_type):
        super(DatasetForCTMR, self).__init__()
        self.root_dir = '/'.join(raw_data_path.split('/')[:-1])
        self.task_type = task_type
        self._load_data()

    def _load_data(self):
        print("Building medical code vocabulary...")
        self._build_feature_voc()
        self.train_data = dill.load(open(os.path.join(self.root_dir, 'downstream', self.task_type, 'train.pkl'), 'rb'))
        self.eval_data = dill.load(open(os.path.join(self.root_dir, 'downstream', self.task_type, 'eval.pkl'), 'rb'))
        self.test_data = dill.load(open(os.path.join(self.root_dir, 'downstream', self.task_type, 'test.pkl'), 'rb'))
        self._build_corr_graph()
        self.train_data = self.patch_data(self.train_data)
        self.eval_data = self.patch_data(self.eval_data)
        self.test_data = self.patch_data(self.test_data)

    def patch_data(self, data):
        new_data = []
        for feat_set, text_note,  label in data:
            feat_set = [self.feature2id[x] for x in feat_set]
            multiview_graph = {'DIAG-DIAG': [], 'MED-MED': [], 'DIAG-MED': []}
            for feat_x in feat_set:
                feat_x_type = 'DIAG' if 'DIAG' in self.id2feature[feat_x] else 'MED'
                for feat_y in feat_set:
                    feat_y_type = 'DIAG' if 'DIAG' in self.id2feature[feat_y] else 'MED'
                    if self.cor_graph.get(feat_x).get(feat_y) > 1:
                        if feat_x_type == feat_y_type:
                            multiview_graph[feat_x_type+'-'+feat_y_type].append((feat_x, feat_y))
                        else:
                            multiview_graph['DIAG-MED'].append((feat_x, feat_y))
            new_data.append((feat_set, multiview_graph, text_note, label))
        return new_data

    def _build_feature_voc(self):
        self.feature2id = json.load(
            open(os.path.join(self.root_dir, 'feature2id.json'), 'r', encoding='utf-8'))
        self.feature2id['[MASK]'] = len(self.feature2id)
        self.id2feature = {v: k for k, v in self.feature2id.items()}

    def _build_corr_graph(self):
        if os.path.exists(os.path.join(self.root_dir, '%s_cooccurrence_graph.pkl'%self.task_type)):
            self.cor_graph = dill.load(open(os.path.join(self.root_dir, '%s_cooccurrence_graph.pkl'%self.task_type), 'rb'))
        else:
            data = self.train_data + self.eval_data + self.test_data
            cor_graph = {}
            for feat_set, _, label in data:
                feat_set = [self.feature2id[x] for x in feat_set]
                for feat_x in feat_set:
                    for feat_y in feat_set:
                        if feat_x not in cor_graph:
                            cor_graph[feat_x] = {}
                        if feat_y not in cor_graph[feat_x]:
                            cor_graph[feat_x][feat_y] = 0
                        cor_graph[feat_x][feat_y] += 1


            self.cor_graph = cor_graph

            dill.dump(self.cor_graph, open(os.path.join(self.root_dir, '%s_cooccurrence_graph.pkl'%self.task_type), 'wb'))

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item):
        return self.train_data[item]





if __name__ == '__main__':
    dataset = DatasetForCTMR('../data/eicu/eicu_data.json', task_type='readmission')
    total_patient = len(dataset)
    total_patient_diag_codes = 0
    total_patient_med_codes = 0
    total_text_length = 0
    diag_set = []
    med_set = []
    for i in range(len(dataset)):
        for d_c in dataset[i][0]:
            if 'DIAG' in dataset.id2feature[d_c]:
                total_patient_diag_codes += 1
                if d_c not in diag_set:
                    diag_set.append(d_c)
            elif 'MED' in dataset.id2feature[d_c]:
                total_patient_med_codes += 1
                if d_c not in med_set:
                    med_set.append(d_c)
        total_text_length += len(dataset[i][2].split(' '))

    print(total_patient)
    print(total_patient_diag_codes/total_patient)
    print(total_patient_med_codes/total_patient)
    print(total_text_length/total_patient)
    print(len(diag_set))
    print(len(med_set))
