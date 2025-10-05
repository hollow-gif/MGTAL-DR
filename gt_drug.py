import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from graph_transformer_layer import GraphTransformerLayer 


class GraphTransformer(nn.Module):
    def __init__(self, device, n_layers, node_input_feature_dim,
                 hidden_dim, out_dim, n_heads, dropout=0.1):
        super(GraphTransformer, self).__init__()
        self.device = device
        self.node_input_feature_dim = node_input_feature_dim
        self.linear_h = nn.Linear(self.node_input_feature_dim, hidden_dim).to(device)
        self.node_perturb = None
        self.layers = nn.ModuleList()  
        for _ in range(n_layers):
            self.layers.append(GraphTransformerLayer(hidden_dim, hidden_dim, 
                                                     n_heads, dropout, 
                                                     layer_norm=True, batch_norm=False, residual=True).to(device))
            
        self.final_projection = nn.Linear(hidden_dim, out_dim).to(device)        
    def get_intermediate_h(self, g):  
        g = g.to(self.device)
        if g.num_nodes() == 0: return torch.empty(0, self.linear_h.out_features, device=self.device)
        if 'drs' not in g.ndata: raise KeyError("Graph g must have 'drs' in ndata for drug GT.")
        h_in = g.ndata['drs'].float().to(self.device)
        if h_in.shape[0] == 0: return torch.empty(0, self.linear_h.out_features, device=self.device)
        if h_in.shape[1] != self.node_input_feature_dim: raise ValueError(f"Drug GT: Input feature dim mismatch. Expected {self.node_input_feature_dim}, got {h_in.shape[1]}")
        h = self.linear_h(h_in)
        return h
        
    def forward(self, g, h_intermediate=None):
        g = g.to(self.device)
        if g.num_nodes() == 0: return torch.empty(0, self.final_projection.out_features, device=self.device)       
        if h_intermediate is None:
            if 'drs' not in g.ndata: raise KeyError("Graph g must have 'drs' in ndata for drug GT forward pass.")
            h_in = g.ndata['drs'].float().to(self.device)
            if h_in.shape[0] == 0: return torch.empty(0, self.final_projection.out_features, device=self.device)
            if h_in.shape[1] != self.node_input_feature_dim: raise ValueError(f"Drug GT: Input feature dim mismatch in forward. Expected {self.node_input_feature_dim}, got {h_in.shape[1]}")
            h = self.linear_h(h_in)
        else:
            h = h_intermediate.to(self.device)
        h_after_perturb = h 
        if self.node_perturb is not None :
            perturb_gpu = self.node_perturb.to(h.device)
            if perturb_gpu.shape == h.shape:
                h_after_perturb = h + perturb_gpu        
        current_h = h_after_perturb  
        for layer in self.layers:
            current_h = layer(g, current_h)     
        last_layer_h = current_h
        final_h = self.final_projection(last_layer_h)
        return final_h
