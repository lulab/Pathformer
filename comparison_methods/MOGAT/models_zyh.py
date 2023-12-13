""" Componets of the model
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
           m.bias.data.fill_(0.0)
           

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903 
    图注意力层
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features   # 节点表示向量的输入特征维度
        self.out_features = out_features   # 节点表示向量的输出特征维度
        self.dropout = dropout    # dropout参数
        self.alpha = alpha     # leakyrelu激活的参数
        self.concat = concat   # 如果为true, 再进行elu激活
        
        # 定义可训练参数，即论文中的W和a
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))  
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # xavier初始化
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)   # xavier初始化
        
        # 定义leakyrelu激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def forward(self, inp, adj):
        """
        inp: input_fea [N, in_features]  in_features表示节点的输入特征向量元素个数
        adj: 图的邻接矩阵 维度[N, N] 非零即一，数据结构基本知识
        """
        h = torch.mm(inp, self.W)   # [N, out_features]
        N = h.size()[0]    # N 图的节点数
        # adj = torch.Tensor(adj)
        adj = adj.to_dense()
        a_input = torch.cat([h.repeat(1, N).view(N*N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2*self.out_features)
        # [N, N, 2*out_features]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        # [N, N, 1] => [N, N] 图注意力的相关系数（未归一化）
        
        zero_vec = -1e12 * torch.ones_like(e)    # 将没有连接的边置为负无穷
        attention = torch.where(adj>0, e, zero_vec)   # [N, N]
        del adj
        # 表示如果邻接矩阵元素大于0时，则两个节点有连接，该位置的注意力系数保留，
        # 否则需要mask并置为非常小的值，原因是softmax的时候这个最小值会不考虑。
        attention = F.softmax(attention, dim=1)    # softmax形状保持不变 [N, N]，得到归一化的注意力权重！
        attention = F.dropout(attention, self.dropout, training=self.training)   # dropout，防止过拟合
        h_prime = torch.matmul(attention, h)  # [N, N].[N, out_features] => [N, out_features]
        # 得到由周围节点通过注意力权重进行更新的表示
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime 
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.MH = nn.ModuleList([
            GraphAttentionLayer(nfeat, nhid, dropout, alpha, concat=True)
            for _ in range(nheads)
        ])
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout, alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)  # (N, nfeat)
        x = torch.cat([head(x, adj) for head in self.MH], dim=1)  # (N, nheads*nhid)
        x = F.dropout(x, self.dropout, training=self.training)  # (N, nheads*nhid)
        x = F.elu(self.out_att(x, adj))
        return x


class Classifier_1(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return x


class VCDN(nn.Module):
    def __init__(self, num_view, num_cls, hvcdn_dim):
        super().__init__()
        self.num_cls = num_cls
        self.model = nn.Sequential(
            nn.Linear(pow(num_cls, num_view), hvcdn_dim),
            nn.LeakyReLU(0.25),
            nn.Linear(hvcdn_dim, num_cls)
        )
        self.model.apply(xavier_init)
        
    def forward(self, in_list):
        num_view = len(in_list)
        for i in range(num_view):
            in_list[i] = torch.sigmoid(in_list[i])
        x = torch.reshape(torch.matmul(in_list[0].unsqueeze(-1), in_list[1].unsqueeze(1)),(-1,pow(self.num_cls,2),1))
        for i in range(2,num_view):
            x = torch.reshape(torch.matmul(x, in_list[i].unsqueeze(1)),(-1,pow(self.num_cls,i+1),1))
        vcdn_feat = torch.reshape(x, (-1,pow(self.num_cls,num_view)))
        output = self.model(vcdn_feat)

        return output

    
def init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hc, gcn_dopout=0.5):
    model_dict = {}
    for i in range(num_view):
        model_dict["E{:}".format(i+1)] = GraphAttentionLayer(dim_list[i], dim_he_list[2], 0.2, gcn_dopout)
        model_dict["C{:}".format(i+1)] = Classifier_1(dim_he_list[2], num_class)
    if num_view >= 2:
        model_dict["C"] = VCDN(num_view, num_class, dim_hc)
    return model_dict


def init_optim(num_view, model_dict, lr_e=1e-4, lr_c=1e-4):
    optim_dict = {}
    for i in range(num_view):
        optim_dict["C{:}".format(i+1)] = torch.optim.Adam(
                list(model_dict["E{:}".format(i+1)].parameters())+list(model_dict["C{:}".format(i+1)].parameters()), 
                lr=lr_e)
    if num_view >= 2:
        optim_dict["C"] = torch.optim.Adam(model_dict["C"].parameters(), lr=lr_c)
    return optim_dict