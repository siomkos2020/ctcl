import torch
import torch.nn as nn
import torch.nn.init as init


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.linear = nn.Linear(input_dim, output_dim)
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.linear.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        # adj: (b, n, n)
        # inp: (b, n, emb_dim)
        support = self.linear(input_feature)
        output = torch.bmm(adjacency, support)
        if self.use_bias:
            output += self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.input_dim) + ' ->' \
               + str(self.output_dim) + ')'


class MultiGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(MultiGCN, self).__init__()
        self.emb = nn.Embedding(input_dim, 64)
        self.emb_drop = nn.Dropout(0.5)
        self.residu_linear1 = nn.Linear(64, hidden_dim)
        self.diag_conv1 = GraphConvolution(64, hidden_dim)
        self.med_conv1 = GraphConvolution(64, hidden_dim)
        self.dm_conv1 = GraphConvolution(64, hidden_dim)
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

        self.output_layer = nn.Linear(hidden_dim, 1)

    def graph_normalized(self, adjacency):
        degree_mat = adjacency.sum(-1)
        degree_mat[degree_mat == 0] = 1e-6
        normed_adj = adjacency / degree_mat.unsqueeze(-1)
        return normed_adj

    def forward(self, input_feature, input_mask, diag_graph, med_graph, dm_graph):
        # input_feature (B, N)
        # input_mask (B, N)
        input_mask = input_mask.unsqueeze(-1)
        input_len = input_mask.sum(1)
        diag_graph = self.graph_normalized(diag_graph)
        med_graph = self.graph_normalized(med_graph)
        dm_graph = self.graph_normalized(dm_graph)

        x = self.emb_drop(self.emb(input_feature))   #(b, n, emb_dim)

        d_x = self.dropout1(self.relu(self.diag_conv1(diag_graph, x)))
        m_x = self.dropout1(self.relu(self.med_conv1(med_graph, x)))
        dm_x = self.dropout1(self.relu(self.dm_conv1(dm_graph, x)))
        f_x = torch.cat([d_x, m_x, dm_x], dim=-1)
        x = self.norm1(self.linear1(f_x) + self.residu_linear1(x))

        d_x = self.dropout2(self.relu(self.diag_conv2(diag_graph, x)))
        m_x = self.dropout2(self.relu(self.med_conv2(med_graph, x)))
        dm_x = self.dropout2(self.relu(self.dm_conv2(dm_graph, x)))
        f_x = torch.cat([d_x, m_x, dm_x], dim=-1)
        x = self.norm2(self.linear2(f_x) + x)

        x = (x * input_mask).sum(1) / input_len   #(b, emb_dim)
        x = self.output_layer(x)
        return torch.sigmoid(x)


class TwoLayerGCNEnc(nn.Module):
    def __init__(self, input_dim, d_model):
        super(TwoLayerGCNEnc, self).__init__()
        self.emb = nn.Embedding(input_dim, 64)
        self.conv1 = GraphConvolution(64, 64)
        self.conv2 = GraphConvolution(64, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.output_layer = nn.Sequential(
            nn.Linear(64, d_model)
        )

    def graph_normalized(self, adjacency):
        degree_mat = adjacency.sum(-1)
        degree_mat[degree_mat == 0] = 1e-4
        normed_adj = adjacency / degree_mat.unsqueeze(-1)
        return normed_adj

    def forward(self, adjacency, input_feature, input_mask):
        # input_feature (B, N)
        # input_mask (B, N)
        input_mask = input_mask.unsqueeze(-1)
        input_len = input_mask.sum(1)
        adjacency = self.graph_normalized(adjacency)
        x = self.dropout(self.emb(input_feature))   #(b, n, emb_dim)
        x = self.conv1(adjacency, x)
        x = self.dropout(self.relu(x)) + x
        x = self.conv2(adjacency, x)
        x = self.dropout(self.relu(x)) + x
        x = (x * input_mask).sum(1) / input_len   #(b, emb_dim)
        x = self.output_layer(self.relu(x))
        return x


class TwoLayerGCNEncWOPOOL(nn.Module):
    def __init__(self, input_dim, d_model):
        super(TwoLayerGCNEncWOPOOL, self).__init__()
        self.emb = nn.Embedding(input_dim, 64)
        self.conv1 = GraphConvolution(64, 64)
        self.conv2 = GraphConvolution(64, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.output_layer = nn.Sequential(
            nn.Linear(64, d_model)
        )

    def graph_normalized(self, adjacency):
        degree_mat = adjacency.sum(-1)
        degree_mat[degree_mat == 0] = 1e-4
        normed_adj = adjacency / degree_mat.unsqueeze(-1)
        return normed_adj

    def forward(self, adjacency, input_feature, input_mask):
        # input_feature (B, N)
        # input_mask (B, N)
        input_mask = input_mask.unsqueeze(-1)
        input_len = input_mask.sum(1)
        adjacency = self.graph_normalized(adjacency)
        x = self.dropout(self.emb(input_feature))   #(b, n, emb_dim)
        x = self.conv1(adjacency, x)
        x = self.dropout(self.relu(x)) + x
        x = self.conv2(adjacency, x)
        x = self.dropout(self.relu(x)) + x
        x = self.output_layer(x)
        return x


class TwoLayerGCN(nn.Module):
    def __init__(self, input_dim):
        super(TwoLayerGCN, self).__init__()
        self.emb = nn.Embedding(input_dim, 64)
        self.conv1 = GraphConvolution(64, 64)
        self.conv2 = GraphConvolution(64, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.output_layer = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def graph_normalized(self, adjacency):
        degree_mat = adjacency.sum(-1)
        degree_mat[degree_mat == 0] = 1e-4
        normed_adj = adjacency / degree_mat.unsqueeze(-1)
        return normed_adj

    def forward(self, adjacency, input_feature, input_mask):
        # input_feature (B, N)
        # input_mask (B, N)
        input_mask = input_mask.unsqueeze(-1)
        input_len = input_mask.sum(1)
        adjacency = self.graph_normalized(adjacency)
        x = self.dropout(self.emb(input_feature))   #(b, n, emb_dim)
        x = self.conv1(adjacency, x)
        x = self.dropout(self.relu(x)) + x
        x = self.conv2(adjacency, x)
        x = self.dropout(self.relu(x)) + x
        x = (x * input_mask).sum(1) / input_len   #(b, emb_dim)
        x = self.output_layer(self.relu(x))
        return torch.sigmoid(x)