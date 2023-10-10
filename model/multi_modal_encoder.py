import torch
import torch.nn as nn
from model.gcn import GraphConvolution, TwoLayerGCNEncWOPOOL
from transformers import AutoModel
import numpy as np


# tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
# model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
#
# model.eval()
# device = torch.device("cuda:1")
# model.to(torch.device("cuda:1"))
# sentence = ["I love mimi-iii", "I love eicu"]
# sentences = tokenizer(sentence, padding=True, truncation=True, max_length=32, return_tensors='pt')
# tokens_tensor = sentences['input_ids'].to(device)
# segments_tensor = sentences['token_type_ids'].to(device)
# attention_mask_ids_tensors = sentences['attention_mask'].to(device)
# print(sentences)
# # input_ids = torch.tensor(sentences['input_ids']).to(torch.device("cuda:1")).unsqueeze(0)
# print(sentences)
# output = model(tokens_tensor, segments_tensor, attention_mask_ids_tensors)
# sequence_output = output.last_hidden_state
# # sequence_output, pooled_output = output
# print(sequence_output.shape)
# print(output)

class MultiGCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(MultiGCNEncoder, self).__init__()
        self.emb = nn.Embedding(input_dim, hidden_dim)
        self.emb_drop = nn.Dropout(0.5)
        self.diag_conv1 = GraphConvolution(hidden_dim, hidden_dim)
        self.med_conv1 = GraphConvolution(hidden_dim, hidden_dim)
        self.dm_conv1 = GraphConvolution(hidden_dim, hidden_dim)
        self.linear1 = nn.Linear(hidden_dim*3, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.diag_conv2 = GraphConvolution(hidden_dim, hidden_dim)
        self.med_conv2 = GraphConvolution(hidden_dim, hidden_dim)
        self.dm_conv2 = GraphConvolution(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim * 3, hidden_dim)
        self.relu = nn.ReLU()
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(0.1)


    def graph_normalized(self, adjacency):
        degree_mat = adjacency.sum(-1)
        degree_mat[degree_mat == 0] = 1e-6
        normed_adj = adjacency / degree_mat.unsqueeze(-1)
        return normed_adj


    def forward(self, input_feature, input_mask, diag_graph, med_graph, dm_graph, return_all=False):
        # input_feature (B, N)
        # input_mask (B, N)
        # return (B, N, hidden_dim)
        diag_graph = self.graph_normalized(diag_graph)
        med_graph = self.graph_normalized(med_graph)
        dm_graph = self.graph_normalized(dm_graph)

        x = self.emb_drop(self.emb(input_feature))   #(b, n, emb_dim)

        d_x = self.dropout1(self.relu(self.diag_conv1(diag_graph, x)))
        m_x = self.dropout1(self.relu(self.med_conv1(med_graph, x)))
        dm_x = self.dropout1(self.relu(self.dm_conv1(dm_graph, x)))
        #
        # d_x =
        # f_x = torch.cat([d_x, m_x, dm_x], dim=-1)
        # x = self.norm1(self.linear1(f_x) + x)
        # x = self.linear1(f_x)

        d_x = self.dropout2(self.relu(self.diag_conv2(diag_graph, d_x)))
        m_x = self.dropout2(self.relu(self.med_conv2(med_graph, m_x)))
        dm_x = self.dropout2(self.relu(self.dm_conv2(dm_graph, dm_x)))
        f_x = torch.cat([d_x, m_x, dm_x], dim=-1)
        x = self.linear2(f_x)
        # x = self.linear2(f_x)

        if return_all:
            return x, d_x, m_x, dm_x
        else:
            return x

class SelfAttn(nn.Module):
    def __init__(self, d_model, nhead=2, hidden_dim=2048):
        super(SelfAttn, self).__init__()
        self.nhead = nhead
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.1, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(0.1)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, x_mask):
        b_size, query_seq, _ = x.size()
        self_mask = x_mask.unsqueeze(1).unsqueeze(1).repeat(1, self.nhead, query_seq, 1)
        self_mask = self_mask.view(b_size*self.nhead, query_seq, query_seq)

        self_x = self.self_attn(x, x, x, attn_mask=self_mask, need_weights=False)[0]
        x = self.norm1(self_x + x)
        x = self.norm2(x + self.ffn(x))
        return x


class CrossModalEncoder(nn.Module):
    def __init__(self, d_model, nhead=2, hidden_dim=2048):
        super(CrossModalEncoder, self).__init__()
        self.nhead = nhead
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.1, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(0.1)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, query, key, key_mask):
        b_size, query_seq, _ = query.size()
        _, key_seq, _ = key.size()

        q2k_mask = key_mask.unsqueeze(1).unsqueeze(1).repeat(1, self.nhead, query_seq, 1)
        q2k_mask = q2k_mask.view(b_size*self.nhead, query_seq, key_seq)

        x, cross_attn = self.cross_attn(query, key, key, attn_mask=q2k_mask, need_weights=True)
        x = self.norm1(x + query)
        x = self.norm2(x + self.ffn(x))

        return x, cross_attn

class MultiModalEncoderWOV(nn.Module):
    def __init__(self,
                 input_dim,
                 device,
                 d_model=256,
                 nhead = 2,
                 modality='full',
                 bert_config="emilyalsentzer/Bio_ClinicalBERT"):
        super(MultiModalEncoderWOV, self).__init__()
        self.device = device
        self.nhead = nhead
        self.modality = modality
        self.d_model = d_model
        self.input_dim = input_dim
        if modality in ['full', 'graph']:
            # self.graph_enc = MultiGCNEncoder(input_dim=input_dim, hidden_dim=d_model)
            self.graph_enc = TwoLayerGCNEncWOPOOL(input_dim=input_dim, d_model=d_model)
            self.graph_dropout = nn.Dropout(0.1)
            self.graph_cross_enc = CrossModalEncoder(d_model=d_model, hidden_dim=2*d_model)
            self.graph_self_enc = SelfAttn(d_model=d_model, hidden_dim=2*d_model)

        if modality in ['full', 'text']:
            self.text_enc =  AutoModel.from_pretrained(bert_config)
            self.text_fcn = nn.Linear(768, d_model)
            self.text_drop_out = nn.Dropout(0.1)

        if modality == 'full':
            self.text_cross_enc = CrossModalEncoder(d_model=d_model, hidden_dim=2*d_model)
            self.text_self_enc = SelfAttn(d_model=d_model, hidden_dim=2*d_model)

    def reset_parameters(self):
        # self.graph_enc = MultiGCNEncoder(input_dim=self.input_dim, hidden_dim=self.d_model)
        self.graph_cross_enc = CrossModalEncoder(d_model=self.d_model, hidden_dim=2*self.d_model)
        self.graph_self_enc = SelfAttn(d_model=self.d_model, hidden_dim=2*self.d_model)
        self.text_cross_enc = CrossModalEncoder(d_model=self.d_model, hidden_dim=2 * self.d_model)
        self.text_self_enc = SelfAttn(d_model=self.d_model, hidden_dim=2 * self.d_model)
        # self.text_fcn = nn.Linear(768, self.d_model)

    def get_graph_encoddings(self, input_feature, input_mask, diag_graph, med_graph, dm_graph, return_all=False):
        graph = diag_graph + med_graph + dm_graph
        graph_mask_zeros = torch.zeros(input_mask.size(), device=self.device)
        graph_mask_zeros = torch.where(input_mask == 0, 1., graph_mask_zeros.double()).float()
        return self.graph_enc(graph, input_feature, graph_mask_zeros)
        # return self.graph_enc(input_feature, input_mask, diag_graph, med_graph, dm_graph, return_all)


    def get_text_encoddings(self, inputs):
        # inputs = tokenizer(input_sentences, padding=True, truncation=True, max_length=32, return_tensors='pt')
        tokens_tensor = inputs['input_ids'].to(self.device)
        segments_tensor = inputs['token_type_ids'].to(self.device)
        attention_mask_ids_tensors = inputs['attention_mask'].to(self.device)
        text_mask = torch.full(attention_mask_ids_tensors.size(), -1e9).to(self.device)
        text_mask = torch.where(attention_mask_ids_tensors.float() == 1, 0., text_mask.double())
        text_mask = text_mask.float()
        if self.modality == 'full':
            with torch.no_grad():
                output = self.text_enc(tokens_tensor, segments_tensor, attention_mask_ids_tensors)
                output = output.last_hidden_state
        else:
            output = self.text_enc(tokens_tensor, segments_tensor, attention_mask_ids_tensors)
            output = output.last_hidden_state

        output = self.text_drop_out(self.text_fcn(output))

        return output, text_mask


    def forward(self, input_nodes, input_mask, diag_graph, med_graph, dm_graph, input_sentences, return_all=False):
        self.device = input_nodes.device

        graph_x = self.get_graph_encoddings(input_nodes, input_mask, diag_graph, med_graph, dm_graph)
        if self.modality in ['full', 'text']:
            text_x, text_mask = self.get_text_encoddings(input_sentences)
            # text_x = self.text_self_enc(text_x, text_mask)

        if self.modality == 'full':
            graph_x = self.graph_cross_enc(graph_x, text_x, text_mask)
            graph_x = self.graph_self_enc(graph_x, input_mask)

            text_x = self.text_cross_enc(text_x, graph_x, input_mask)
            text_x = self.text_self_enc(text_x, text_mask)

            if return_all:
                return graph_x, None, None, None, text_x
            else:
                return graph_x, text_x
        elif self.modality == 'graph':
            return graph_x, None
        elif self.modality == 'text':
            return None, text_x




class MultiModalEncoder(nn.Module):
    def __init__(self,
                 input_dim,
                 device,
                 d_model=256,
                 nhead = 2,
                 modality='full',
                 bert_config="emilyalsentzer/Bio_ClinicalBERT"):
        super(MultiModalEncoder, self).__init__()
        self.device = device
        self.nhead = nhead
        self.modality = modality
        self.d_model = d_model
        self.input_dim = input_dim
        if modality in ['full', 'graph']:
            self.graph_enc = MultiGCNEncoder(input_dim=input_dim, hidden_dim=d_model)
            # self.graph_enc = TwoLayerGCNEnc(input_dim=input_dim, d_model=d_model)
            self.graph_dropout = nn.Dropout(0.1)
            self.graph_cross_enc = CrossModalEncoder(d_model=d_model, hidden_dim=2*d_model)
            self.graph_self_enc = SelfAttn(d_model=d_model, hidden_dim=2*d_model)

        if modality in ['full', 'text']:
            self.text_enc =  AutoModel.from_pretrained(bert_config)
            self.text_fcn = nn.Linear(768, d_model)
            self.text_drop_out = nn.Dropout(0.1)

        if modality == 'full':
            self.text_cross_enc = CrossModalEncoder(d_model=d_model, hidden_dim=2*d_model)
            self.text_self_enc = SelfAttn(d_model=d_model, hidden_dim=2*d_model)

    def reset_parameters(self):
        # self.graph_enc = MultiGCNEncoder(input_dim=self.input_dim, hidden_dim=self.d_model)
        self.graph_cross_enc = CrossModalEncoder(d_model=self.d_model, hidden_dim=2*self.d_model)
        self.graph_self_enc = SelfAttn(d_model=self.d_model, hidden_dim=2*self.d_model)
        self.text_cross_enc = CrossModalEncoder(d_model=self.d_model, hidden_dim=2 * self.d_model)
        self.text_self_enc = SelfAttn(d_model=self.d_model, hidden_dim=2 * self.d_model)
        # self.text_fcn = nn.Linear(768, self.d_model)

    def get_graph_encoddings(self, input_feature, input_mask, diag_graph, med_graph, dm_graph, return_all=False):
        # graph = diag_graph + med_graph + dm_graph
        # graph_mask_zeros = torch.zeros(input_mask.size(), device=self.device)
        # graph_mask_zeros = torch.where(input_mask == 0, 1., graph_mask_zeros.double()).float()
        # return self.graph_enc(graph, input_feature, graph_mask_zeros)
        return self.graph_enc(input_feature, input_mask, diag_graph, med_graph, dm_graph, return_all)


    def get_text_encoddings(self, inputs):
        # inputs = tokenizer(input_sentences, padding=True, truncation=True, max_length=32, return_tensors='pt')
        tokens_tensor = inputs['input_ids'].to(self.device)
        segments_tensor = inputs['token_type_ids'].to(self.device)
        attention_mask_ids_tensors = inputs['attention_mask'].to(self.device)
        text_mask = torch.full(attention_mask_ids_tensors.size(), -1e9).to(self.device)
        text_mask = torch.where(attention_mask_ids_tensors.float() == 1, 0., text_mask.double())
        text_mask = text_mask.float()
        if self.modality == 'full':
            with torch.no_grad():
                output = self.text_enc(tokens_tensor, segments_tensor, attention_mask_ids_tensors)
                output = output.last_hidden_state
        else:
            output = self.text_enc(tokens_tensor, segments_tensor, attention_mask_ids_tensors)
            output = output.last_hidden_state

        output = self.text_drop_out(self.text_fcn(output))

        return output, text_mask


    def forward(self, input_nodes, input_mask, diag_graph, med_graph, dm_graph, input_sentences, return_all=False):
        self.device = input_nodes.device
        if self.modality in ['full', 'graph']:
            if return_all:
                graph_x, d_x, m_x, dm_x = \
                    self.get_graph_encoddings(input_nodes, input_mask, diag_graph, med_graph, dm_graph, return_all=True)
            else:
                graph_x = self.get_graph_encoddings(input_nodes, input_mask, diag_graph, med_graph, dm_graph)
        if self.modality in ['full', 'text']:
            text_x, text_mask = self.get_text_encoddings(input_sentences)
            # text_x = self.text_self_enc(text_x, text_mask)

        if self.modality == 'full':
            graph_x, gt_attn = self.graph_cross_enc(graph_x, text_x, text_mask)
            graph_x = self.graph_self_enc(graph_x, input_mask)

            text_x, text_attn = self.text_cross_enc(text_x, graph_x, input_mask)
            text_x = self.text_self_enc(text_x, text_mask)

            if return_all:
                return graph_x, d_x, m_x, dm_x, text_x, gt_attn, text_attn
            else:
                return graph_x, text_x, gt_attn, text_attn
        elif self.modality == 'graph':
            return graph_x, None
        elif self.modality == 'text':
            return None, text_x


class MultiModalForClassification(nn.Module):
    def __init__(self,
                 input_dim,
                 device,
                 cross_mode='concat',
                 num_label=1,
                 d_model=256,
                 modality='full',
                 nhead=2,
                 bert_config="../emilyalsentzer/Bio_ClinicalBERT"
                 ):
        super(MultiModalForClassification, self).__init__()
        self.device = device
        self.cross_mode = cross_mode
        self.modality = modality
        self.multi_model_enc = MultiModalEncoder(
            input_dim,
            device,
            modality=modality,
            nhead=nhead,
            d_model=d_model,
            bert_config=bert_config
        )
        # self.multi_model_enc = MultiModalEncoderWOV(
        #     input_dim,
        #     device,
        #     modality=modality,
        #     nhead=nhead,
        #     d_model=d_model,
        #     bert_config=bert_config
        # )
        if self.modality == 'full':
            if cross_mode == 'concat':
                # self.out_norm = nn.LayerNorm(d_model*2)
                self.output_layer = nn.Linear(d_model*2, num_label)
            elif cross_mode == 'sum':
                self.o1 = nn.Linear(d_model, num_label)
                self.o2 = nn.Linear(d_model, num_label)
                self.o2w = nn.Sequential(
                    nn.Linear(2*d_model, d_model),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(d_model, 1)
                )
            elif cross_mode == 'co':
                print('cross mode', cross_mode)
                self.output_layer = nn.Linear(d_model, num_label)
        elif self.modality == 'graph':
            self.out = nn.Linear(d_model, num_label)
        elif self.modality == 'text':
            self.out = nn.Linear(d_model, num_label)

    def from_pretrained(self, model_state_dict):
        self.multi_model_enc.load_state_dict(model_state_dict)

    def get_contrastive_loss(self, graph_enc, text_enc, d_x, m_x):
        # batch_size = graph_enc.size()[0]
        # sim_mat = torch.mm(graph_enc, text_enc.transpose(0, 1))
        # sim_mat = torch.softmax(sim_mat, dim=-1)
        # sim_mat = sim_mat[torch.arange(batch_size), torch.arange(batch_size)]
        # loss = -torch.log(sim_mat).mean()

        # cross_rep = torch.cat([graph_enc, text_enc], dim=-1)
        # sim_mat = torch.bmm(cross_rep, cross_rep.transpose(1, 2))
        sim_mat = torch.bmm(graph_enc, text_enc.transpose(1, 2))
        sim_mat = torch.softmax(sim_mat, dim=-1)
        sim_mat = sim_mat[:, 0, 0]
        loss1 = -torch.log(sim_mat).mean()

        view_sim1 = torch.softmax(torch.mm(d_x, m_x.transpose(0, 1))*0.1, dim=-1)[:, 0]
        loss3 = -torch.log(view_sim1).mean()

        return 0.1*loss3 + 0.09 * loss1
        # return 0.1 * loss3 + 0.01 * loss1
        # return loss1 #+ 0.1*loss3


    def forward(self, input_nodes, input_mask, diag_graph, med_graph, dm_graph, input_sentences, tokenizer=None):
        if tokenizer is not None:
            batch_size, seq_num = input_nodes.size()
            n_num = 4
            repeat_input_nodes = []
            repeat_input_mask = []
            repeat_diag_graph = []
            repeat_med_graph = []
            repeat_dm_graph = []
            repeat_input_sentences = []
            for i in range(batch_size):
                neg_ids = [i]
                while len(neg_ids) < n_num:
                    rand_i = np.random.randint(0, batch_size)
                    if rand_i not in neg_ids:
                        neg_ids.append(rand_i)
                repeat_input_nodes.append(torch.cat([input_nodes[j].unsqueeze(0) for j in neg_ids], dim=0))
                repeat_input_mask.append(torch.cat([input_mask[j].unsqueeze(0) for j in neg_ids], dim=0))
                repeat_diag_graph.append(torch.cat([diag_graph[j].unsqueeze(0) for j in neg_ids], dim=0))
                repeat_med_graph.append(torch.cat([med_graph[j].unsqueeze(0) for j in neg_ids], dim=0))
                repeat_dm_graph.append(torch.cat([dm_graph[j].unsqueeze(0) for j in neg_ids], dim=0))
                repeat_input_sentences.extend([input_sentences[j] for j in neg_ids])

            input_nodes = torch.cat(repeat_input_nodes).view(batch_size*n_num, -1)
            input_mask = torch.cat(repeat_input_mask).view(batch_size*n_num, -1)
            diag_graph = torch.cat(repeat_diag_graph).view(batch_size*n_num, seq_num, seq_num)
            med_graph = torch.cat(repeat_med_graph).view(batch_size*n_num, seq_num, seq_num)
            dm_graph = torch.cat(repeat_dm_graph).view(batch_size*n_num, seq_num, seq_num)
            input_sentences = tokenizer(repeat_input_sentences, padding=True, truncation=True, max_length=256, return_tensors='pt')

        if tokenizer is not None:
            graph_encs, d_x, m_x, dm_x, text_encs, gt_attn, text_attn = self.multi_model_enc(
                input_nodes, input_mask, diag_graph, med_graph, dm_graph, input_sentences, return_all=True
            )
        else:
            graph_encs, text_encs, gt_attn, text_attn = self.multi_model_enc(
                input_nodes, input_mask, diag_graph, med_graph, dm_graph, input_sentences
            )

        # graph_encs = graph_encs.view(batch_size, n_num, seq_num, -1)
        # text_encs = text_encs.view(batch_size, n_num, text_seq_size, -1)
        if self.modality == 'full':
            graph_mask_zeros = torch.zeros(input_mask.size(), device=self.device)
            graph_mask_zeros = torch.where(input_mask == 0, 1., graph_mask_zeros.double()).float()
            graph_mask_zeros = graph_mask_zeros.unsqueeze(-1)
            input_len = graph_mask_zeros.sum(1)
            graph_encs = (graph_encs * graph_mask_zeros).sum(1) / input_len  # (b, emb_dim)
            text_encs = text_encs[:, 0, :]
            if tokenizer is not None:
                graph_encs = graph_encs.view(batch_size, n_num, -1)
                text_encs = text_encs.view(batch_size, n_num, -1)

                # d_x = torch.cat([d_x, dm_x], dim=-1)
                # m_x = torch.cat([m_x, dm_x], dim=-1)

                d_x = (d_x * graph_mask_zeros).sum(1) / input_len  # (b, emb_dim)
                m_x = (m_x * graph_mask_zeros).sum(1) / input_len  # (b, emb_dim)
                # dm_x = (dm_x * graph_mask_zeros).sum(1) / input_len  # (b, emb_dim)

                d_x = d_x.view(batch_size, n_num, -1)[:, 0, :]
                m_x = m_x.view(batch_size, n_num, -1)[:, 0, :]
                # dm_x = dm_x.view(batch_size, n_num, -1)[:, 0, :]


            if self.cross_mode == 'concat':
                # x = self.out_norm(torch.cat([graph_encs, text_encs], dim=-1))
                if tokenizer is not None:
                    cont_loss = self.get_contrastive_loss(graph_encs, text_encs, d_x, m_x)
                else:
                    cont_loss = None
                if self.training and tokenizer is not None:
                    x = torch.cat([graph_encs[:, 0, :], text_encs[:, 0, :]], dim=-1)
                else:
                    x = torch.cat([graph_encs, text_encs], dim=-1)
                y = self.output_layer(x)
                return torch.sigmoid(y), cont_loss, gt_attn, text_attn
            elif self.cross_mode == 'sum':
                o1 = torch.sigmoid(self.o1(graph_encs))
                o2 = torch.sigmoid(self.o2(text_encs))
                ow = torch.sigmoid(self.o2w(torch.cat([graph_encs, text_encs], dim=-1)))
                y = ow * o1 + (1-ow) * o2
                return y
            elif self.cross_mode == 'co':
                return torch.sigmoid(self.output_layer(graph_encs))
        elif self.modality == 'graph':
            graph_mask_zeros = torch.zeros(input_mask.size(), device=self.device)
            graph_mask_zeros = torch.where(input_mask == 0, 1., graph_mask_zeros.double()).float()
            graph_mask_zeros = graph_mask_zeros.unsqueeze(-1)
            input_len = graph_mask_zeros.sum(1)
            graph_encs = (graph_encs * graph_mask_zeros).sum(1) / input_len  # (b, emb_dim)
            return torch.sigmoid(self.out(graph_encs))
        else:
            text_encs = text_encs[:, 0, :]
            return torch.sigmoid(self.out(text_encs))


class MultiModalForRegression(nn.Module):
    def __init__(self,
                 input_dim,
                 device,
                 cross_mode='concat',
                 num_label=1,
                 d_model=256,
                 modality='full',
                 nhead=2,
                 bert_config="../emilyalsentzer/Bio_ClinicalBERT"
                 ):
        super(MultiModalForRegression, self).__init__()
        self.device = device
        self.cross_mode = cross_mode
        self.modality = modality
        self.multi_model_enc = MultiModalEncoder(
            input_dim,
            device,
            modality=modality,
            nhead=nhead,
            d_model=d_model,
            bert_config=bert_config
        )
        # self.multi_model_enc = MultiModalEncoderWOV(
        #     input_dim,
        #     device,
        #     modality=modality,
        #     nhead=nhead,
        #     d_model=d_model,
        #     bert_config=bert_config
        # )
        if self.modality == 'full':
            if cross_mode == 'concat':
                # self.out_norm = nn.LayerNorm(d_model*2)
                self.output_layer = nn.Linear(d_model*2, num_label, bias=True)
            elif cross_mode == 'sum':
                self.o1 = nn.Linear(d_model, num_label)
                self.o2 = nn.Linear(d_model, num_label)
                self.o2w = nn.Sequential(
                    nn.Linear(2*d_model, d_model),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(d_model, 1)
                )
            elif cross_mode == 'co':
                print('cross mode', cross_mode)
                self.output_layer = nn.Linear(d_model, num_label)
        elif self.modality == 'graph':
            self.out = nn.Linear(d_model, num_label, bias=True)
        elif self.modality == 'text':
            self.out = nn.Linear(d_model, num_label, bias=True)

    def from_pretrained(self, model_state_dict):
        self.multi_model_enc.load_state_dict(model_state_dict)

    def get_contrastive_loss(self, graph_enc, text_enc, d_x, m_x):
        # batch_size = graph_enc.size()[0]
        # sim_mat = torch.mm(graph_enc, text_enc.transpose(0, 1))
        # sim_mat = torch.softmax(sim_mat, dim=-1)
        # sim_mat = sim_mat[torch.arange(batch_size), torch.arange(batch_size)]
        # loss = -torch.log(sim_mat).mean()

        # cross_rep = torch.cat([graph_enc, text_enc], dim=-1)
        # sim_mat = torch.bmm(cross_rep, cross_rep.transpose(1, 2))
        sim_mat = torch.bmm(graph_enc, text_enc.transpose(1, 2))
        sim_mat = torch.softmax(sim_mat, dim=-1)
        sim_mat = sim_mat[:, 0, 0]
        loss1 = -torch.log(sim_mat).mean()

        view_sim1 = torch.softmax(torch.mm(d_x, m_x.transpose(0, 1))*0.1, dim=-1)[:, 0]
        loss3 = -torch.log(view_sim1).mean()

        return 0.1*loss3 + 0.09 * loss1
        # return 0.1 * loss3 + 0.01 * loss1
        # return loss1 #+ 0.1*loss3


    def forward(self, input_nodes, input_mask, diag_graph, med_graph, dm_graph, input_sentences, tokenizer=None):
        if tokenizer is not None:
            batch_size, seq_num = input_nodes.size()
            n_num = 4
            repeat_input_nodes = []
            repeat_input_mask = []
            repeat_diag_graph = []
            repeat_med_graph = []
            repeat_dm_graph = []
            repeat_input_sentences = []
            for i in range(batch_size):
                neg_ids = [i]
                while len(neg_ids) < n_num:
                    rand_i = np.random.randint(0, batch_size)
                    if rand_i not in neg_ids:
                        neg_ids.append(rand_i)
                repeat_input_nodes.append(torch.cat([input_nodes[j].unsqueeze(0) for j in neg_ids], dim=0))
                repeat_input_mask.append(torch.cat([input_mask[j].unsqueeze(0) for j in neg_ids], dim=0))
                repeat_diag_graph.append(torch.cat([diag_graph[j].unsqueeze(0) for j in neg_ids], dim=0))
                repeat_med_graph.append(torch.cat([med_graph[j].unsqueeze(0) for j in neg_ids], dim=0))
                repeat_dm_graph.append(torch.cat([dm_graph[j].unsqueeze(0) for j in neg_ids], dim=0))
                repeat_input_sentences.extend([input_sentences[j] for j in neg_ids])

            input_nodes = torch.cat(repeat_input_nodes).view(batch_size*n_num, -1)
            input_mask = torch.cat(repeat_input_mask).view(batch_size*n_num, -1)
            diag_graph = torch.cat(repeat_diag_graph).view(batch_size*n_num, seq_num, seq_num)
            med_graph = torch.cat(repeat_med_graph).view(batch_size*n_num, seq_num, seq_num)
            dm_graph = torch.cat(repeat_dm_graph).view(batch_size*n_num, seq_num, seq_num)
            input_sentences = tokenizer(repeat_input_sentences, padding=True, truncation=True, max_length=256, return_tensors='pt')

        if tokenizer is not None:
            graph_encs, d_x, m_x, dm_x, text_encs, gt_attn, text_attn = self.multi_model_enc(
                input_nodes, input_mask, diag_graph, med_graph, dm_graph, input_sentences, return_all=True
            )
        else:
            graph_encs, text_encs, gt_attn, text_attn = self.multi_model_enc(
                input_nodes, input_mask, diag_graph, med_graph, dm_graph, input_sentences
            )

        # graph_encs = graph_encs.view(batch_size, n_num, seq_num, -1)
        # text_encs = text_encs.view(batch_size, n_num, text_seq_size, -1)
        if self.modality == 'full':
            graph_mask_zeros = torch.zeros(input_mask.size(), device=self.device)
            graph_mask_zeros = torch.where(input_mask == 0, 1., graph_mask_zeros.double()).float()
            graph_mask_zeros = graph_mask_zeros.unsqueeze(-1)
            input_len = graph_mask_zeros.sum(1)
            graph_encs = (graph_encs * graph_mask_zeros).sum(1) / input_len  # (b, emb_dim)
            text_encs = text_encs[:, 0, :]
            if tokenizer is not None:
                graph_encs = graph_encs.view(batch_size, n_num, -1)
                text_encs = text_encs.view(batch_size, n_num, -1)

                # d_x = torch.cat([d_x, dm_x], dim=-1)
                # m_x = torch.cat([m_x, dm_x], dim=-1)

                d_x = (d_x * graph_mask_zeros).sum(1) / input_len  # (b, emb_dim)
                m_x = (m_x * graph_mask_zeros).sum(1) / input_len  # (b, emb_dim)
                # dm_x = (dm_x * graph_mask_zeros).sum(1) / input_len  # (b, emb_dim)

                d_x = d_x.view(batch_size, n_num, -1)[:, 0, :]
                m_x = m_x.view(batch_size, n_num, -1)[:, 0, :]
                # dm_x = dm_x.view(batch_size, n_num, -1)[:, 0, :]


            if self.cross_mode == 'concat':
                # x = self.out_norm(torch.cat([graph_encs, text_encs], dim=-1))
                if tokenizer is not None:
                    cont_loss = self.get_contrastive_loss(graph_encs, text_encs, d_x, m_x)
                else:
                    cont_loss = None
                if self.training and tokenizer is not None:
                    x = torch.cat([graph_encs[:, 0, :], text_encs[:, 0, :]], dim=-1)
                else:
                    x = torch.cat([graph_encs, text_encs], dim=-1)
                y = self.output_layer(x)
                return y, cont_loss, gt_attn, text_attn
            elif self.cross_mode == 'sum':
                o1 = torch.sigmoid(self.o1(graph_encs))
                o2 = torch.sigmoid(self.o2(text_encs))
                ow = torch.sigmoid(self.o2w(torch.cat([graph_encs, text_encs], dim=-1)))
                y = ow * o1 + (1-ow) * o2
                return y
            elif self.cross_mode == 'co':
                return self.output_layer(graph_encs)
        elif self.modality == 'graph':
            graph_mask_zeros = torch.zeros(input_mask.size(), device=self.device)
            graph_mask_zeros = torch.where(input_mask == 0, 1., graph_mask_zeros.double()).float()
            graph_mask_zeros = graph_mask_zeros.unsqueeze(-1)
            input_len = graph_mask_zeros.sum(1)
            graph_encs = (graph_encs * graph_mask_zeros).sum(1) / input_len  # (b, emb_dim)
            return self.out(graph_encs)
        else:
            text_encs = text_encs[:, 0, :]
            return self.out(text_encs)



class MultiModalPretraining(nn.Module):
    def __init__(self,
                 input_dim,
                 device,
                 diag_voc_size=0,
                 med_voc_size=0,
                 cross_mode='concat',
                 d_model=256,
                 nhead=2,
                 bert_config="emilyalsentzer/Bio_ClinicalBERT"
                 ):
        super(MultiModalPretraining, self).__init__()
        self.device = device
        self.cross_mode = cross_mode
        self.multi_model_enc = MultiModalEncoder(
            input_dim,
            device,
            nhead=nhead,
            d_model=d_model,
            bert_config=bert_config
        )
        self.text_pred_code = nn.Linear(d_model, input_dim)
        self.text_pred_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(10.))
        self.med_pred_header = nn.Linear(2*d_model, diag_voc_size)
        self.med_pred_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(10.))
        self.diag_pred_header = nn.Linear(2 * d_model, med_voc_size)
        self.diag_pred_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(10.))

    def get_text_pred_code_loss(self, input_sentences, code_labels):
        text_encs, _ = self.multi_model_enc.get_text_encoddings(input_sentences)
        text_encs = text_encs[:, 0, :]
        output = self.text_pred_code(text_encs)
        code_pred = torch.sigmoid(output)
        # print(code_pred)
        # print(code_labels.shape)
        # exit()
        loss = self.text_pred_loss(output, code_labels.float())

        code_pred_numpy = code_pred.detach().cpu().numpy()
        code_labels_numpy = code_labels.cpu().numpy()
        # print(code_pred_numpy[np.arange(code_labels_numpy.shape[0]), code_labels_numpy])

        code_pred_numpy[code_pred_numpy > 0.5] = 1
        code_pred_numpy[code_pred_numpy <= 0.5] = 0
        jac = (code_pred_numpy * code_labels_numpy).sum() / ((code_pred_numpy.sum() + code_labels_numpy.sum()-
                                                            (code_pred_numpy * code_labels_numpy).sum()))

        return loss, jac

    def get_med_pred_diag_loss(self, batch_feat_ids, batch_feat_mask, batch_diag_graph, batch_med_graph,
                               batch_dm_graph, batch_labels, batch_text):
        graph_encs, text_encs = self.multi_model_enc(
            batch_feat_ids, batch_feat_mask, batch_diag_graph, batch_med_graph, batch_dm_graph, batch_text
        )
        graph_mask_zeros = torch.zeros(batch_feat_mask.size(), device=graph_encs.device)
        graph_mask_zeros = torch.where(batch_feat_mask == 0, 1., graph_mask_zeros.double()).float()
        graph_mask_zeros = graph_mask_zeros.unsqueeze(-1)
        input_len = graph_mask_zeros.sum(1)
        graph_encs = (graph_encs * graph_mask_zeros).sum(1) / (input_len + 1e-6)  # (b, emb_dim)
        text_encs = text_encs[:, 0, :]
        label_pred = self.med_pred_header(torch.cat([graph_encs, text_encs], dim=-1))
        label_pred = torch.sigmoid(label_pred)
        loss = self.med_pred_loss(label_pred, batch_labels)

        label_pred_numpy = label_pred.detach().cpu().numpy()
        label_tg_numpy = batch_labels.detach().cpu().numpy()
        label_pred_numpy[label_pred_numpy > 0.5] = 1
        label_pred_numpy[label_pred_numpy <= 0.5] = 0
        jac = (label_pred_numpy * label_tg_numpy).sum() / ((label_pred_numpy.sum() + label_tg_numpy.sum()-
                                                            (label_pred_numpy * label_tg_numpy).sum()))
        return loss, jac

    def get_diag_pred_med_loss(self, batch_feat_ids, batch_feat_mask, batch_diag_graph, batch_med_graph,
                               batch_dm_graph, batch_labels, batch_text):
        graph_encs, text_encs = self.multi_model_enc(
            batch_feat_ids, batch_feat_mask, batch_diag_graph, batch_med_graph, batch_dm_graph, batch_text
        )
        graph_mask_zeros = torch.zeros(batch_feat_mask.size(), device=graph_encs.device)
        graph_mask_zeros = torch.where(batch_feat_mask == 0, 1., graph_mask_zeros.double()).float()
        graph_mask_zeros = graph_mask_zeros.unsqueeze(-1)
        input_len = graph_mask_zeros.sum(1)
        graph_encs = (graph_encs * graph_mask_zeros).sum(1) / (input_len + 1e-6)  # (b, emb_dim)
        text_encs = text_encs[:, 0, :]
        label_pred = self.diag_pred_header(torch.cat([graph_encs, text_encs], dim=-1))
        label_pred = torch.sigmoid(label_pred)
        loss = self.diag_pred_loss(label_pred, batch_labels)

        label_pred_numpy = label_pred.detach().cpu().numpy()
        label_tg_numpy = batch_labels.detach().cpu().numpy()
        label_pred_numpy[label_pred_numpy > 0.5] = 1
        label_pred_numpy[label_pred_numpy <= 0.5] = 0
        jac = (label_pred_numpy * label_tg_numpy).sum() / ((label_pred_numpy.sum() + label_tg_numpy.sum()-
                                                            (label_pred_numpy * label_tg_numpy).sum()))
        return loss, jac




    def forward(self, input_nodes, input_mask, diag_graph, med_graph, dm_graph, input_sentences, code_labels, code_mask_pos):

        text_encs, _ = self.multi_model_enc.get_text_encoddings(input_sentences)

        text_encs = text_encs[:, 0, :]
        # cross_modal_rep = graph_encs
        code_mask_loss, code_mask_acc = self.get_masked_code_loss(text_encs, code_labels)
        return code_mask_loss, code_mask_acc






