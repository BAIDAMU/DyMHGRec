import torch
import torch.nn as nn
import torch.nn.functional as F
from layer_monmery1 import EmbeddingLayer, hypergraph_part, one_hypergraph, TransitionLayer, DotProductAttention
import numpy as np
import math


class HypDrug(nn.Module):
    def __init__(
        self,
        vocab_size,
        ehr_adj,
        ddi_adj,
        ddi_mask_H,
        emb_dim=256,
        device=torch.device("cpu:0"),
    ):
        super(HypDrug, self).__init__()

        self.device = device
        self.emb_dim = emb_dim
        self.ehr_adj = ehr_adj

        # pre-embedding
        self.diag_embedding_layer = EmbeddingLayer(vocab_size[0], emb_dim)
        self.proc_embedding_layer = EmbeddingLayer(vocab_size[1], emb_dim)
        self.med_embedding_layer = EmbeddingLayer(vocab_size[2],emb_dim)
        
        
        #多通道超图编码器
        self.hypergraph_encode = hypergraph_part(emb_dim, emb_dim, device=device)
        #单通道超图编码器
        self.one_hypergraph_encode = one_hypergraph(emb_dim, emb_dim, device=device)
        #上下文感知
        self.content_aware = TransitionLayer(vocab_size[2], emb_dim, emb_dim, emb_dim, emb_dim)

        self.attention = DotProductAttention(emb_dim, emb_dim)
        self.encoders = nn.ModuleList(
            [nn.GRU(emb_dim, emb_dim, batch_first=True) for _ in range(3)]
        )
        self.ddi_gcn = GCN(
            voc_size=vocab_size[2], emb_dim=emb_dim, adj=ddi_adj, device=device
        )
        self.ehr_gcn = GCN(
            voc_size=vocab_size[2], emb_dim=emb_dim, adj=ehr_adj, device=device
        )
        self.classifer = nn.Linear(emb_dim, 1)
        self.proj = nn.Linear(2 * vocab_size[2], vocab_size[2])
        # graphs, bipartite matrix
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.tensor_ddi_mask_H = torch.FloatTensor(ddi_mask_H).to(device)

        self.mapping = nn.Sequential(nn.Linear(3*emb_dim, emb_dim), nn.ReLU(), nn.Linear(emb_dim, 131))

    def forward(self, input, seq_divided):

        # patient health representation
        diag_seq = []
        proc_seq = []
        med_seq = []

        d_embeddings = self.diag_embedding_layer()
        p_embeddings = self.proc_embedding_layer()
        m_embeddings = self.med_embedding_layer()
        unrelated_embeddings = self.med_embedding_layer()
        
        for adm_num, admission in enumerate(input):     
            d_seq = admission[0]
            p_seq = admission[1]
            
            i1, i2 = self.hypergraph_encode(d_seq, p_seq, d_embeddings, p_embeddings)
            diag_seq.append(i1)
            proc_seq.append(i2)

        for idx, adm in enumerate(input):
            if len(input) <= 1 or idx==0:
                med_1 = torch.zeros((1, 1, self.emb_dim)).to(self.device)
            else:
                adm[2] = input[idx - 1][2][:]
                m_seq = adm[2]
                med_1 = self.one_hypergraph_encode(m_seq, m_embeddings)
                
            med_seq.append(med_1)

        diag_seq = torch.cat(diag_seq, dim=1)  # (1,seq,dim)
        proc_seq = torch.cat(proc_seq, dim=1)  # (1,seq,dim)# (1,seq,dim)
        med_seq = torch.cat(med_seq, dim=1)
        
        o1, h1 = self.encoders[0](diag_seq)
        o2, h2 = self.encoders[1](proc_seq)
        o3, h3 = self.encoders[2](med_seq)

        o1 = o1.squeeze(dim=0).squeeze(dim=0)
        o2 = o2.squeeze(dim=0).squeeze(dim=0)
        o3 = o3.squeeze(dim=0).squeeze(dim=0)


        #历史药物
        if len(input) > 1 :
            history_content_rep = []
            hidden_state = None
            for idx, history_divided in enumerate(seq_divided[:-1]):
                divided_med = history_divided[2]
                med_h, hidden_state = self.content_aware(m_embeddings, divided_med, self.ehr_gcn(), unrelated_embeddings, hidden_state)
                history_content_rep.append(med_h)
            history_rep = self.attention(torch.vstack(history_content_rep))
        
        query = torch.cat([o1, o2, o3], dim=-1).squeeze(0)
        if query.dim() == 1:
            query = query.unsqueeze(0)
        query = query[-1:]
        out = self.mapping(query)
        
        if len(input) > 1 :
            weigthed_v = torch.diag(history_rep)
            history_aware = torch.mm(self.ddi_gcn(), weigthed_v)
            history_aware = self.classifer(history_aware).t()
            h = torch.cat([out, history_aware], -1)
            out = self.proj(h)

        neg_pred_prob = F.sigmoid(out)
        neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)

        batch_neg = F.sigmoid(neg_pred_prob.mul(self.tensor_ddi_adj)).sum()
        return out, batch_neg
        
class GCN(nn.Module):
    def __init__(
        self, voc_size, emb_dim, adj, dropout_rate=0.3, device=torch.device("cpu:0")
    ):
        super().__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device

        adj = self.normalize(adj + np.eye(adj.shape[0]))

        self.adj = torch.FloatTensor(adj).to(device)
        self.x = torch.eye(voc_size).to(device)

        self.gcn1 = GraphConvolution(voc_size, emb_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)

    def forward(self):
        node_embedding = self.gcn1(self.x, self.adj)
        node_embedding = F.relu(node_embedding)
        node_embedding = self.dropout(node_embedding)
        node_embedding = self.gcn2(node_embedding, self.adj)
        return node_embedding

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.0
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )
