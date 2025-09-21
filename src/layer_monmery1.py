import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv


class SingleHeadAttentionLayer(nn.Module):
    def __init__(self, query_size, key_size, value_size, attention_size):
        super().__init__()
        self.attention_size = attention_size
        self.dense_q = nn.Linear(query_size, attention_size)
        self.dense_k = nn.Linear(key_size, attention_size)
        self.dense_v = nn.Linear(query_size, value_size)

    def forward(self, q, k, v):
        query = self.dense_q(q)
        key = self.dense_k(k)
        value = self.dense_v(v)
        g = torch.div(torch.matmul(query, key.T), math.sqrt(self.attention_size))
        score = torch.softmax(g, dim=-1)
        output = torch.sum(torch.unsqueeze(score, dim=-1) * value, dim=-2)
        return output

#诊断、手术、药物嵌入层
class EmbeddingLayer(nn.Module):
    def __init__(self, code_num, code_size):
        super().__init__()
        self.code_num = code_num
        self.c_embeddings = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(code_num, code_size)))
        
    def forward(self):
        return self.c_embeddings

class hypergraph_part(nn.Module):
    def __init__(self, in_channels, out_channels, device):
        super().__init__()
        self.device = device
        self.out_channels = out_channels
        self.conv = HypergraphConv(in_channels, out_channels)
        self.conv_gat = HypergraphConv(in_channels, out_channels, use_attention=True , attention_mode='node')
        self.linear_layer = nn.Linear(in_features=2*out_channels, out_features=out_channels, bias=False)#######128->in_features修改为2*32 out_features修改为32

    def forward(self, c_it, medicine_it, c_embeddings, m_embeddings):
        # diagnosis channel 将一次就诊中所有的diagnoses形成一个超图
        diagnosis_hyperedge_index = []
        diagnosis_hyperedge_index.append(list(range(0, len(c_it))))
        edge = len(c_it) * [0]
        diagnosis_hyperedge_index.append(edge)
        diagnosis_hyperedge_index = torch.tensor(diagnosis_hyperedge_index).to(c_embeddings.device)
        dia_embedding = c_embeddings[c_it]
        dia_node_feature = self.conv(dia_embedding, diagnosis_hyperedge_index)#根据当前诊断超图的超边结构通过超图卷积层进行诊断节点的特征更新
        # medicine channel 将一次就诊中所有的medicine形成一个超图
        medicine_hyperedge_index = []
        medicine_hyperedge_index.append(list(range(0, len(medicine_it))))
        edge = len(medicine_it) * [0]
        medicine_hyperedge_index.append(edge)
        medicine_hyperedge_index = torch.tensor(medicine_hyperedge_index).to(c_embeddings.device)
        med_embedding = m_embeddings[medicine_it]
        med_node_feature = self.conv(med_embedding, medicine_hyperedge_index)

        # diagnosis-medicine channel 将一次就诊中的diangoses与medicine共现关系形成一个超图
        dia_med_hyperedges, dia_med_emb, dia_med_hyperedge_attr = dual_hypergraphs(c_it, medicine_it, c_embeddings,
                                                                               m_embeddings)
        dia_med_hyperedges = torch.tensor(dia_med_hyperedges).to(c_embeddings.device)
        dia_med_emb = dia_med_emb.to(c_embeddings.device)
        dia_med_hyperedge_attr = dia_med_hyperedge_attr.to(c_embeddings.device)
        dia_med_node_feature = self.conv_gat(dia_med_emb, dia_med_hyperedges, hyperedge_attr=dia_med_hyperedge_attr) #, hyperedge_attr=dia_med_hyperedge_attr
        disease_final_rep, medicine_final_rep, final_representation = concat_representations(dia_med_node_feature, dia_node_feature, med_node_feature, c_it, medicine_it)
        final_representation = self.linear_layer(final_representation)
        i1, i2 = self.resi_output(final_representation, c_it, medicine_it)
        return i1, i2
    #残差连接
    def resi_output(self, representation, disease_index, medicine_index):
        embedding_trained = [torch.zeros([1, self.out_channels], dtype=torch.float).to(self.device) for _ in range(2)]

        i = 0
        for d in disease_index:
            embedding_trained[0] += representation[i]
            i += 1
        for m in medicine_index:
            embedding_trained[1] += representation[i]
            i += 1
            
        i1 = embedding_trained[0]
        i2 = embedding_trained[1]
        return i1.unsqueeze(0), i2.unsqueeze(0)

class one_hypergraph(nn.Module):
    def __init__(self, in_channels, out_channels, device):
        super().__init__()
        self.device = device
        self.out_channels = out_channels
        self.conv = HypergraphConv(in_channels, out_channels)
    
    def forward(self, medicine_it, m_embeddings):
        # medicine channel 将一次就诊中所有的medicine形成一个超图
        medicine_hyperedge_index = []
        medicine_hyperedge_index.append(list(range(0, len(medicine_it))))
        edge = len(medicine_it) * [0]
        medicine_hyperedge_index.append(edge)
        medicine_hyperedge_index = torch.tensor(medicine_hyperedge_index).to(self.device)
        med_embedding = m_embeddings[medicine_it]
        med_node_feature = self.conv(med_embedding, medicine_hyperedge_index)
        
        med = self.output(med_node_feature, medicine_it)
        return med

    def output(self, representation, medicine_index):
        embedding_trained = torch.zeros([1, self.out_channels], dtype=torch.float).to(self.device)
        
        i = 0
        for m in medicine_index:
            embedding_trained += representation[i]
            i += 1
            
        i1 = embedding_trained
        
        return i1.unsqueeze(0)       

def dual_hypergraphs(c_it, med_it, c_embeddings, m_embeddings):
    # Get indices of non-zero elements (i.e., diagnosed diseases and prescribed medications)
    disease_index = list(range(0, len(c_it)))
    med_index = [x+len(disease_index) for x in range(0, len(med_it))]

    # Extract embeddings for diseases and medications
    disease_embeddings = c_embeddings[c_it]
    med_embeddings = m_embeddings[med_it]
    # Combine disease and medication embeddings
    combined_embeddings = torch.cat((disease_embeddings, med_embeddings), dim=0)
    # Create hyperedges
    hyperedges = []
    indicate_edges = []
    count = 0
    hyper_edge_index = []
    for disease in disease_index:
        hyperedges = hyperedges + [disease] + med_index
        indicate_edges = indicate_edges + (len(med_index)+1)*[count]
        count = count+1
        # Initialize hyperedge attributes as learnable parameters
    hyper_edge_index.append(hyperedges)
    hyper_edge_index.append(indicate_edges)
    edge_num = max(hyper_edge_index[1]) + 1
    hyperedge_attr = nn.Parameter(torch.randn(edge_num, combined_embeddings.shape[1]))
    return hyper_edge_index, combined_embeddings, hyperedge_attr


def concat_representations(dia_med_node_feature, dia_node_feature, med_node_feature, c_it, med_it):
    # Get the number of diseases and medicines from the input tensors
    num_diseases = len(c_it)
    num_meds = len(med_it)
    # Assuming each disease in dia_med_node_feature has a representation combined with each medicine
    # We'll average these combined representations to get a single representation for each disease and medicine
    dia_med_disease_rep = dia_med_node_feature[:num_diseases]
    dia_med_medicine_rep = dia_med_node_feature[num_diseases:]
    # Concatenate the representations
    disease_final_rep = torch.cat([dia_med_disease_rep, dia_node_feature], dim=1)
    medicine_final_rep = torch.cat([dia_med_medicine_rep, med_node_feature], dim=1)
    final_representation = torch.cat([disease_final_rep, medicine_final_rep], dim=0)
    return disease_final_rep, medicine_final_rep, final_representation

        
class TransitionLayer(nn.Module):
    def __init__(self, code_num, graph_size, hidden_size, t_attention_size, t_output_size):
        super().__init__()
        self.gru = nn.GRUCell(input_size=graph_size, hidden_size=hidden_size)
        self.single_head_attention = SingleHeadAttentionLayer(graph_size, graph_size, t_output_size, t_attention_size)
        self.activation = nn.Tanh()

        self.code_num = code_num
        self.hidden_size = hidden_size

    def forward(self, m_embeddings, divided, ddi_embeddings, unrelated_embeddings, hidden_state=None):
        m1, m2, m3 = divided[:, 0], divided[:, 1], divided[:, 2]
        m1_index = torch.where(m1 > 0)[0]
        m2_index = torch.where(m2 > 0)[0]
        m3_index = torch.where(m3 > 0)[0]
        h_new = torch.zeros((self.code_num, self.hidden_size), dtype=m_embeddings.dtype).to(m_embeddings.device)
        output_m1 = 0
        output_m23 = 0
        if len(m1_index) > 0: #长期性疾病
            m1_embedding = m_embeddings[m1_index]
            h = hidden_state[m1_index] if hidden_state is not None else None
            h_m1 = self.gru(m1_embedding, h)
            h_new[m1_index] = h_m1
            output_m1, _ = torch.max(h_m1, dim=-2)
        if len(m2_index) + len(m3_index) > 0: #突发性疾病
            q = torch.vstack([ddi_embeddings[m2_index], unrelated_embeddings[m3_index]])
            v = torch.vstack([m_embeddings[m2_index], m_embeddings[m3_index]])
            h_m23 = self.activation(self.single_head_attention(q, q, v))
            h_new[m2_index] = h_m23[:len(m2_index)]
            h_new[m3_index] = h_m23[len(m2_index):]
            output_m23, _ = torch.max(h_m23, dim=-2)
        if len(m1_index) == 0:
            output = output_m23
        elif len(m2_index) + len(m3_index) == 0:
            output = output_m1
        else:
            output, _ = torch.max(torch.vstack([output_m1, output_m23]), dim=-2)
        return output, h_new

class DotProductAttention(nn.Module):
    def __init__(self, value_size, attention_size):
        super().__init__()
        self.attention_size = attention_size
        self.context = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(attention_size, 1)))
        self.dense = nn.Linear(value_size, attention_size)

    def forward(self, x):
        t = self.dense(x)
        vu = torch.matmul(t, self.context).squeeze()
        score = torch.softmax(vu, dim=-1)
        output = torch.sum(x * torch.unsqueeze(score, dim=-1), dim=-2)
        return output