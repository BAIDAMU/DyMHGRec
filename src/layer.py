import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv
from torch.nn.parameter import Parameter

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
class one_hypergraph(nn.Module):
    def __init__(self, in_channels, out_channels, device):
        super().__init__()
        self.device = device
        self.out_channels = out_channels
        self.conv = HypergraphConv(in_channels, out_channels)
    
    def forward(self, medicine_it, m_embeddings,pretrained_model):
        # medicine channel 将一次就诊中所有的medicine形成一个超图
        medicine_hyperedge_index = []
        medicine_hyperedge_index.append(list(range(0, len(medicine_it))))
        edge = len(medicine_it) * [0]
        medicine_hyperedge_index.append(edge)
        medicine_hyperedge_index = torch.tensor(medicine_hyperedge_index).to(self.device)
        med_embedding = m_embeddings[medicine_it]  # 需要定义m_embeddings
        med_node_feature = self.conv(med_embedding, medicine_hyperedge_index)
        
        med = self.output(med_node_feature, pretrained_model, medicine_it)
        return med

    def output(self, representation, pretrained_embedding, medicine_index):
        embedding_pretrained = torch.zeros([1, self.out_channels], dtype=torch.float).to(self.device) 
        embedding_trained = torch.zeros([1, self.out_channels], dtype=torch.float).to(self.device)
        
        i = 0
        for m in medicine_index:
            # embedding_pretrained[1] += pretrained_embedding[1].weight.data[m]
            embedding_pretrained += pretrained_embedding['embeddings.2.weight'][m]
            embedding_trained += representation[i]
            i += 1
            
        i1 = embedding_pretrained + embedding_trained
        return i1.unsqueeze(0)
 
class hypergraph_part(nn.Module):
    def __init__(self, in_channels, out_channels, device):
        super().__init__()
        self.device = device
        self.out_channels = out_channels
        self.conv = HypergraphConv(in_channels, out_channels)
        self.conv_gat = HypergraphConv(in_channels, out_channels, use_attention=True , attention_mode='node')
        self.linear_layer = nn.Linear(in_features=128, out_features=64, bias=False)

    def forward(self, c_it, medicine_it, c_embeddings, m_embeddings,pretrain_model):
        # diagnosis channel 将一次就诊中所有的diagnoses形成一个超图
        diagnosis_hyperedge_index = []
        diagnosis_hyperedge_index.append(list(range(0, len(c_it))))
        edge = len(c_it) * [0]
        diagnosis_hyperedge_index.append(edge)
        diagnosis_hyperedge_index = torch.tensor(diagnosis_hyperedge_index).to(c_embeddings.device)
        dia_embedding = c_embeddings[c_it]#取出当前就诊中所有诊断的嵌入
        dia_node_feature = self.conv(dia_embedding, diagnosis_hyperedge_index)#根据当前诊断超图的超边结构通过超图卷积层进行诊断节点的特征更新
        # medicine channel 将一次就诊中所有的medicine形成一个超图
        medicine_hyperedge_index = []
        medicine_hyperedge_index.append(list(range(0, len(medicine_it))))
        edge = len(medicine_it) * [0]
        medicine_hyperedge_index.append(edge)
        medicine_hyperedge_index = torch.tensor(medicine_hyperedge_index).to(c_embeddings.device)
        med_embedding = m_embeddings[medicine_it]  # 需要定义m_embeddings
        med_node_feature = self.conv(med_embedding, medicine_hyperedge_index)

        # diagnosis-medicine channel 将一次就诊中的diangoses与medicine共现关系形成一个超图
        dia_med_hyperedges, dia_med_emb, dia_med_hyperedge_attr = dual_hypergraphs(c_it, medicine_it, c_embeddings,
                                                                               m_embeddings)
        dia_med_hyperedges = torch.tensor(dia_med_hyperedges).to(c_embeddings.device)
        dia_med_emb = dia_med_emb.to(c_embeddings.device)
        dia_med_hyperedge_attr = dia_med_hyperedge_attr.to(c_embeddings.device)
        dia_med_node_feature = self.conv_gat(dia_med_emb, dia_med_hyperedges, hyperedge_attr=dia_med_hyperedge_attr) #, hyperedge_attr=dia_med_hyperedge_attr
        #dia_med_node_feature = self.conv(dia_med_emb, dia_med_hyperedges)
        disease_final_rep, medicine_final_rep, final_representation = concat_representations(dia_med_node_feature, dia_node_feature, med_node_feature, c_it, medicine_it)
        final_representation = self.linear_layer(final_representation)
        i1, i2 = self.resi_output(final_representation, pretrain_model, c_it, medicine_it)
        # return dia_node_feature #返回哪个通道的表示
        # return final_representation
        return i1, i2
    #残差连接
    def resi_output(self, representation, pretrained_embedding, disease_index, medicine_index):
        embedding_pretrained = [torch.zeros([1, self.out_channels], dtype=torch.float).to(self.device) for _ in range(2)]
        embedding_trained = [torch.zeros([1, self.out_channels], dtype=torch.float).to(self.device) for _ in range(2)]
        
        i = 0
        for d in disease_index:
            # embedding_pretrained[0] += pretrained_embedding[0].weight.data[d]
            embedding_pretrained[0] += pretrained_embedding['embeddings.0.weight'][d]
            embedding_trained[0] += representation[i]
            i += 1
        for m in medicine_index:
            # embedding_pretrained[1] += pretrained_embedding[1].weight.data[m]
            embedding_pretrained[1] += pretrained_embedding['embeddings.1.weight'][m]
            embedding_trained[1] += representation[i]
            i += 1
            
        i1 = embedding_pretrained[0] + embedding_trained[0]
        i2 = embedding_pretrained[1] + embedding_trained[1]
        return i1.unsqueeze(0), i2.unsqueeze(0)

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
    #final_representation = torch.cat([dia_node_feature, med_node_feature], dim=0)
    return disease_final_rep, medicine_final_rep, final_representation

class hypergraph_enconder(nn.Module):
    def __init__(self, in_channels, out_channels, device):
        super().__init__()
        self.device = device
        self.out_channels = out_channels
        self.conv = HypergraphConv(in_channels, out_channels)
        self.conv_gat = HypergraphConv(in_channels, out_channels, use_attention=True , attention_mode='node')
        self.linear_layer = nn.Linear(in_features=128, out_features=64, bias=False)

    def forward(self, adm, c_embeddings, p_embeddings, m_embeddings,pretrain_model):
        diag_seq = adm[0]
        proc_seq = adm[1]
        med_seq = adm[2]
          
        # diagnosis channel 将一次就诊中所有的diagnoses形成一个超图
        diagnosis_hyperedge_index = []
        diagnosis_hyperedge_index.append(list(range(0, len(diag_seq))))
        edge = len(diag_seq) * [0]
        diagnosis_hyperedge_index.append(edge)
        diagnosis_hyperedge_index = torch.tensor(diagnosis_hyperedge_index).to(self.device)
        dia_embedding = c_embeddings[diag_seq]#取出当前就诊中所有诊断的嵌入
        dia_node_feature = self.conv(dia_embedding, diagnosis_hyperedge_index)#根据当前诊断超图的超边结构通过超图卷积层进行诊断节点的特征更新
        # dia_node_feature = F.relu(dia_node_feature)
        # medicine channel 将一次就诊中所有的medicine形成一个超图
        procedure_hyperedge_index = []
        procedure_hyperedge_index.append(list(range(0, len(proc_seq))))
        edge = len(proc_seq) * [0]
        procedure_hyperedge_index.append(edge)
        procedure_hyperedge_index = torch.tensor(procedure_hyperedge_index).to(self.device)
        pro_embedding = p_embeddings[proc_seq]  # 需要定义m_embeddings
        pro_node_feature = self.conv(pro_embedding, procedure_hyperedge_index)
        # pro_node_feature = F.relu(pro_node_feature)
        if len(med_seq) > 0 :
            # medicine channel 将一次就诊中所有的medicine形成一个超图
            medicine_hyperedge_index = []
            medicine_hyperedge_index.append(list(range(0, len(med_seq))))
            edge = len(med_seq) * [0]
            medicine_hyperedge_index.append(edge)
            medicine_hyperedge_index = torch.tensor(medicine_hyperedge_index).to(self.device)
            med_embedding = m_embeddings[med_seq]  # 需要定义m_embeddings
            med_node_feature = self.conv(med_embedding, medicine_hyperedge_index)
            # med_node_feature = F.relu(med_node_feature)
            # diagnosis-medicine channel 将一次就诊中的diangoses与medicine共现关系形成一个超图
            dia_med_hyperedges, dia_med_emb, dia_med_hyperedge_attr = dual_hypergraphs(diag_seq, med_seq, c_embeddings, m_embeddings)                                                                               
            dia_med_hyperedges = torch.tensor(dia_med_hyperedges).to(self.device)
            dia_med_emb = dia_med_emb.to(self.device)
            dia_med_hyperedge_attr = dia_med_hyperedge_attr.to(self.device)
            dia_med_node_feature = self.conv_gat(dia_med_emb, dia_med_hyperedges, hyperedge_attr=dia_med_hyperedge_attr) 
            # dia_med_node_feature = F.relu(dia_med_node_feature)
            # procdure-medicine channel 将一次就诊中的procdure与medicine共现关系形成一个图
            pro_med_hyperedges, pro_med_emb, pro_med_hyperedge_attr = dual_hypergraphs(proc_seq, med_seq, p_embeddings, m_embeddings)
            pro_med_hyperedges = torch.tensor(pro_med_hyperedges).to(self.device)
            pro_med_emb = pro_med_emb.to(self.device)
            pro_med_hyperedge_attr = pro_med_hyperedge_attr.to(self.device)
            pro_med_node_feature = self.conv_gat(pro_med_emb, pro_med_hyperedges, hyperedge_attr=pro_med_hyperedge_attr)
            # pro_med_node_feature = F.relu(pro_med_node_feature)
        else:
            med_node_feature = torch.zeros([len(med_seq), self.out_channels], dtype=torch.float).to(self.device)
            dia_med_node_feature = torch.zeros([len(diag_seq), self.out_channels], dtype=torch.float).to(self.device)
            pro_med_node_feature = torch.zeros([len(proc_seq), self.out_channels], dtype=torch.float).to(self.device)
            
                                               
        final_representation = concat_representations(dia_med_node_feature, dia_node_feature, med_node_feature,pro_med_node_feature, pro_node_feature, diag_seq, proc_seq, med_seq)
        # i1, i2, i3 = self.resi_output(final_representation, pretrain_model, adm)
        # return i1, i2, i3
        return final_representation
    #残差连接
    # def resi_output(self, representation, pretrained_embedding, adm):
    #     embedding_pretrained = [torch.zeros([1, self.out_channels], dtype=torch.float).to(self.device) for _ in range(3)]
    #     embedding_trained = [torch.zeros([1, self.out_channels], dtype=torch.float).to(self.device) for _ in range(3)]
        
    #     i = 0
    #     for d in adm[0]:
    #         # embedding_pretrained[0] += pretrained_embedding[0].weight.data[d]
    #         embedding_pretrained[0] += pretrained_embedding['embeddings.0.weight'][d]
    #         embedding_trained[0] += representation[i]
    #         i += 1
    #     for p in adm[1]:
    #         # embedding_pretrained[1] += pretrained_embedding[1].weight.data[p]
    #         embedding_pretrained[1] += pretrained_embedding['embeddings.1.weight'][p]
    #         embedding_trained[1] += representation[i]
    #         i += 1
            
    #     if len(adm[2]) == 0:  # 加上这个模块，padding的是一个嵌入，不加这个模块，padding的是一个全0嵌入
    #         embedding_trained[2] = torch.zeros((1, self.out_channels)).to(self.device)
    #     for m in adm[2]:
    #         # embedding_pretrained[2] += pretrained_embedding[2].weight.data[m]
    #         embedding_pretrained[2] += pretrained_embedding['embeddings.2.weight'][m]
    #         embedding_trained[2] += representation[i]
    #         i += 1
            
    #     i1 = embedding_pretrained[0] + embedding_trained[0]
    #     i2 = embedding_pretrained[1] + embedding_trained[1]
    #     i3 = embedding_pretrained[2] + embedding_trained[2]
    #     return i1.unsqueeze(0), i2.unsqueeze(0), i3.unsqueeze(0)

        

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

def concat_representations1(dia_med_node_feature, dia_node_feature, med_node_feature, pro_med_node_feature, pro_node_feature, c_it, p_it, med_it):

    num_diseases = len(c_it)
    num_meds = len(med_it)
    
    dia_med_disease_rep = dia_med_node_feature[:num_diseases]
    dia_med_medicine_rep = dia_med_node_feature[num_diseases:]

    num_pros = len(p_it)
    pro_med_procedure_rep = pro_med_node_feature[:num_pros]
    pro_med_medicine_rep = pro_med_node_feature[num_pros:]

    disease_final_rep = dia_med_disease_rep + dia_node_feature
    procedure_final_rep = pro_med_procedure_rep + pro_node_feature
    medicine_final_rep = dia_med_medicine_rep + pro_med_medicine_rep + med_node_feature

    final_representation = torch.cat([disease_final_rep, procedure_final_rep, medicine_final_rep], dim=0)
    # return disease_final_rep, medicine_final_rep, final_representation
    return final_representation

class pairgraph_part(nn.Module):
    def __init__(self, code_size, trans_embedding_dim):
        super().__init__()
        self.embedding_dim = trans_embedding_dim #超参
        self.graph_encoder = GraphEncoder(embed_size=code_size, heads=8, depth=3, trans_embedding_dim= trans_embedding_dim)
        #self.transformer_layer = TransformerLayer(embed_size=self.embedding_dim, heads=8)  # Adjust as needed
    def forward(self, c_it, medicine_it, c_embeddings, m_embeddings, pretrained_embedding):
        # Get indices of non-zero elements (i.e., diagnosed diseases and prescribed medications)
        # disease_indices = torch.nonzero(c_it).squeeze().tolist()#[0, 1, 2, 3, 4, 5, 6, 7]
        # if type(disease_indices) is int:
        #     disease_index_list = []
        #     disease_index_list.append(disease_indices)
        #     disease_indices = disease_index_list
        # indices = torch.nonzero(medicine_it).squeeze()
        # if indices.numel() == 1:
        #     med_indices = [indices.tolist()]
        # else:
        #     med_indices = torch.nonzero(medicine_it).squeeze().tolist()#[0, 1, 2]
        
        # Extract embeddings for diseases and medications
        # disease_embeddings = c_embeddings[disease_indices]#取出索引对应的行torch.Size([8, 64]) 表示诊断编码
        # med_embeddings = m_embeddings[med_indices]#torch.Size([13, 256])
        disease_embeddings = c_embeddings[c_it]#取出索引对应的行torch.Size([8, 64]) 表示诊断编码
        med_embeddings = m_embeddings[medicine_it]#torch.Size([13, 256])
        # Combine disease and medication embeddings
        combined_embeddings = torch.cat((disease_embeddings, med_embeddings), dim=0)
        # Construct pairwise graph
        # nodes = torch.cat([c_it, medicine_it])  # Combining diseases and medicines as nodes
        node_embeddings = combined_embeddings  # Assuming self.embedding is the embedding layer for nodes
        # Use transformer to encode the graph
        #encoded_nodes, attention_weights = self.transformer_layer(node_embeddings, node_embeddings, node_embeddings)
        encoded_nodes = self.graph_encoder(node_embeddings)#torch.Size([11, 64])
        # nodes_i1, nodes_i2 = self.resi_output(encoded_nodes, pretrained_embedding, disease_indices, med_indices)
        nodes_i1, nodes_i2 = self.resi_output(encoded_nodes, pretrained_embedding, c_it, medicine_it)
        return nodes_i1, nodes_i2
        # return encoded_nodes
    
    def resi_output(self, representation, pretrained_embedding, disease_index, medicine_index):
        embedding_pretrained = [torch.zeros([1, self.embedding_dim], dtype=torch.float).to(representation.device) for _ in range(2)]
        embedding_trained = [torch.zeros([1, self.embedding_dim], dtype=torch.float).to(representation.device) for _ in range(2)]
        
        i = 0
        for d in disease_index:
            # embedding_pretrained[0] += pretrained_embedding[0].weight.data[d]
            embedding_pretrained[0] += pretrained_embedding['embeddings.0.weight'][d]
            embedding_trained[0] += representation[i]
            i += 1
        for m in medicine_index:
            # embedding_pretrained[1] += pretrained_embedding[1].weight.data[m]
            embedding_pretrained[1] += pretrained_embedding['embeddings.1.weight'][m]
            embedding_trained[1] += representation[i]
            i += 1
            
        i1 = embedding_pretrained[0] + embedding_trained[0]
        i2 = embedding_pretrained[1] + embedding_trained[1]
        return i1.unsqueeze(0), i2.unsqueeze(0)
    
class GraphEncoder(nn.Module):
    def __init__(self, embed_size, heads, depth, trans_embedding_dim):
        super(GraphEncoder, self).__init__()
        self.encoder_layers = nn.ModuleList([
            TransformerBlock(embed_size, heads, trans_embedding_dim) for _ in range(depth)
        ])

    def forward(self, node_embeddings, mask=None):#torch.Size([23, 256])诊断和药物拼接嵌入
        out = node_embeddings
        for layer in self.encoder_layers:
            out = layer(out, out, out, mask)
        return out
 
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads, ouput_emb):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, ouput_emb)

    def forward(self, values, keys, query, mask):
        N, _ = values.size()

        # Reshape
        values = values.view(N, self.heads, -1)
        keys = keys.view(N, self.heads, -1)
        queries = query.view(N, self.heads, -1)

        # Ensure the size of the last dimension is correct
        _, _,  value_len = values.size()
        _, _, key_len = keys.size()
        _, _, query_len = queries.size()

        # Split the last dimension into (num_nodes, head_dim)
        values = values.view(N, -1, self.heads, self.head_dim).transpose(1, 2)
        keys = keys.view(N, -1, self.heads, self.head_dim).transpose(1, 2)
        queries = queries.view(N, -1, self.heads, self.head_dim).transpose(1, 2)

        # Self attention
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])  # (N, heads, query_len, key_len)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.nn.functional.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).transpose(1, 2).contiguous().view(N, -1, self.heads * self.head_dim)

        out = self.fc_out(out)
        out = out.squeeze(1)
        return out
   
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, trans_embedding_dim):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads, trans_embedding_dim)
        self.norm1 = nn.LayerNorm(embed_size)#两个层归一化
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(#前馈神经网络
            nn.Linear(embed_size, trans_embedding_dim),#这个trans_embedding_dim是在第一个linear层变换呢还是在第二个linear层变换呢！！！
            nn.ReLU(),
            nn.Linear(trans_embedding_dim, trans_embedding_dim) #256
        )

    def forward(self, value, key, query, mask):#value, key, query均表示诊断和药物拼接节点嵌入
        attention = self.attention(value, key, query, mask)#自注意层 加权

        # Add skip connection, run through normalization and finally feed forward network
        out = self.norm1(attention + query)#torch.Size([23, 256]) 加权注意力和查询向量相加 残差连接 归一化处理
        out = self.feed_forward(out)#torch.Size([23, 256]) 
        out = self.norm2(out + attention)#第二次残差连接和归一化
        return out #torch.Size([23, 256]

class MedicalCodeAttention(nn.Module):
    def __init__(self, hidden_size, attention_dim):
        super(MedicalCodeAttention, self).__init__()
        self.dense = nn.Linear(hidden_size, attention_dim)
        self.context = nn.Parameter(nn.init.xavier_uniform_(torch.empty(attention_dim, 1)))

    def forward(self, x):
        # x: [node_num, hidden_size]
        attention_weights = torch.tanh(self.dense(x))  # [node_num, attention_dim]
        attention_weights = torch.matmul(attention_weights, self.context).squeeze()  # [node_num]
        attention_weights = F.softmax(attention_weights, dim=-1)
        output = torch.sum(x * attention_weights.unsqueeze(-1), dim=0)  # [hidden_size]
        return output

class VisitRepresentation(nn.Module):
    def __init__(self, hidden_size, attention_dim, output_dim):
        super(VisitRepresentation, self).__init__()
        self.hypergraph_attention = MedicalCodeAttention(hidden_size, hidden_size)#attention_dim
        self.pairgraph_attention = MedicalCodeAttention(hidden_size, attention_dim)
        self.dense = nn.Linear(2 * hidden_size, output_dim) #256*4=1024

    def forward(self, hypergraph_repre):
        hypergraph_weighted = self.hypergraph_attention(hypergraph_repre) #这是计算hypergraph_repre中节点之间的重要性
        # pairgraph_weighted = self.pairgraph_attention(pair_repre) #这是计算pair_repre中节点之间的重要性
        #combined = torch.cat([hypergraph_weighted, pairgraph_weighted, transgraph_repre], dim=-1) #这是将hypergraph_repre得到的就诊表示与pair_repre得到的就诊表示cat起来
        # combined = torch.cat([hypergraph_weighted, pairgraph_weighted], dim=-1) #这是将hypergraph_repre得到的就诊表示与pair_repre得到的就诊表示cat起来
        # output = self.dense(combined) #这边将最终的就诊表示接入一个dense层，这一步总感觉,本来combined维度很大，然后突然变成了一个256维度，信息丢失太多吧;试一下不加这个dense与加这个dense的效果
        # return combined #torch.Size([128])
        return hypergraph_weighted.unsqueeze(0)

class visit_DotProductAttention(nn.Module):
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


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
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