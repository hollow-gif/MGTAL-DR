import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import numpy as np
import scipy.sparse as sp 

def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field:(edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True)}
    return func

def scaled_exp(field, scale_constant):
    def func(edges):
        return {field: torch.exp((edges.data[field] / scale_constant).clamp(-5, 5))}
    return func

class MultiHeadAttentionLayer(nn.Module):
   
    def __init__(self, in_dim, out_dim, num_heads):
        super(MultiHeadAttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)

    def propagate_attention(self, g):
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'))
        g.apply_edges(scaled_exp('score', np.sqrt(self.out_dim)))
        eids = g.edges()
        g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
        g.send_and_recv(eids, fn.copy_edge('score', 'score'), fn.sum('score', 'z'))
    
    def forward(self, g, h):
        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)
        g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)
        self.propagate_attention(g)
        head_out = g.ndata['wV'] / (g.ndata['z'] + torch.full_like(g.ndata['z'], 1e-6))
        for key in ['Q_h', 'K_h', 'V_h', 'wV', 'z']:
            if key in g.ndata: del g.ndata[key]
        if 'score' in g.edata: del g.edata['score']
        if 'V_h' in g.edata: del g.edata['V_h']
        return head_out

class GraphTransformerLayer(nn.Module):
    
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.1, layer_norm=False, batch_norm=True, residual=True, scales=[1,2,4]):
        super(GraphTransformerLayer, self).__init__()
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.residual = residual
        self.scales = scales     
        self.attention_heads = nn.ModuleList([
            MultiHeadAttentionLayer(in_dim, out_dim // num_heads, num_heads)
            for _ in scales
        ])     
        self.scale_weight_generator = nn.Sequential(
            nn.Linear(in_dim, max(16, len(scales) * 4)), 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(16, len(scales) * 4), len(scales))
        )
        self.linear_O = nn.Linear(out_dim, out_dim, bias=False)

        if self.layer_norm:
            self.layer_norm1 = nn.LayerNorm(out_dim)
            self.layer_norm2 = nn.LayerNorm(out_dim)
        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(out_dim)
            self.batch_norm2 = nn.BatchNorm1d(out_dim)

        self.FFN_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_layer2 = nn.Linear(out_dim * 2, out_dim)
        
        if in_dim != out_dim and residual:
            self.residual_proj1 = nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.residual_proj1 = None

    @staticmethod
    def get_diffusion_graph(g, t, k_neighbor=20):
       
        if g.num_nodes() == 0:
            return dgl.graph(([], []), num_nodes=0)      
        adj = g.adjacency_matrix(scipy_fmt="coo") + g.adjacency_matrix(scipy_fmt="coo").T 
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        norm_adj_sp = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
        norm_adj = torch.from_numpy(norm_adj_sp.toarray()).float().to(g.device)
        diffused_adj = torch.matrix_power(norm_adj, t)
        num_nodes = g.num_nodes()
        diffused_adj.fill_diagonal_(0)
        actual_k = min(k_neighbor, num_nodes - 1)
        if actual_k <= 0:
            return dgl.graph(([], []), num_nodes=num_nodes)
        top_k_vals, top_k_indices = torch.topk(diffused_adj, k=actual_k, dim=1)    
        src_nodes = torch.arange(num_nodes, device=g.device).unsqueeze(1).expand(-1, actual_k).flatten()
        dst_nodes = top_k_indices.flatten()
        new_g = dgl.graph((src_nodes, dst_nodes), num_nodes=num_nodes)
        return new_g

    def forward(self, g, h):
        h_in1 = h
        scale_outputs = []
        for i, t_step in enumerate(self.scales):
            diffused_g = self.get_diffusion_graph(g, t_step, k_neighbor=20) 
            attn_out = self.attention_heads[i](diffused_g, h) 
            scale_outputs.append(attn_out)
        scale_weights = F.softmax(self.scale_weight_generator(h), dim=1) 
        stacked_scale_outputs = torch.stack(scale_outputs, dim=1) 
        adaptive_weights = scale_weights.unsqueeze(-1).unsqueeze(-1)
        attn_out = (adaptive_weights * stacked_scale_outputs).sum(dim=1)
        h_att = attn_out.view(-1, self.out_channels)
        h_O = F.dropout(self.linear_O(h_att), self.dropout, training=self.training)            
        if self.residual:
            h_res1 = (self.residual_proj1(h_in1) if self.residual_proj1 else h_in1) + h_O
        else:
            h_res1 = h_O        
        h_norm1 = self.layer_norm1(h_res1) if self.layer_norm else \
                  (self.batch_norm1(h_res1) if self.batch_norm and h_res1.dim() > 1 and h_res1.shape[0] > 0 else h_res1)   
        h_in2 = h_norm1
        h_ffn = self.FFN_layer1(h_norm1)
        h_ffn = F.relu(h_ffn)
        h_ffn = F.dropout(h_ffn, self.dropout, training=self.training)
        h_ffn = self.FFN_layer2(h_ffn)
        if self.residual:
            h_res2 = h_in2 + h_ffn
        else:
            h_res2 = h_ffn        
        h_final = self.layer_norm2(h_res2) if self.layer_norm else \
                  (self.batch_norm2(h_res2) if self.batch_norm and h_res2.dim() > 1 and h_res2.shape[0] > 0 else h_res2)
        return h_final        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.num_heads, self.residual)
