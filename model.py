import dgl
import dgl.nn.pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F 
import gt_drug 
import gt_disease 

_device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

class ViewAttention(nn.Module):
   
    def __init__(self, in_dim, hid_dim=128):
        super(ViewAttention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, 1, bias=False)
        )
    
    def forward(self, z_list):
        valid_z_list = [z for z in z_list if z is not None and z.numel() > 0]
        if not valid_z_list:
            return None
            
        z_stack = torch.stack(valid_z_list, dim=1)
        w = self.project(z_stack)
        beta = torch.softmax(w, dim=1)
        return (beta * z_stack).sum(1)

class FiLMLayer(nn.Module):
    def __init__(self, in_dim, context_dim):
        super().__init__()
        self.context_proj = nn.Linear(context_dim, in_dim * 2)

    def forward(self, h, context):
        if h is None or h.numel() == 0:
            return h
        gamma, beta = self.context_proj(context).chunk(2, dim=-1)
        return gamma.unsqueeze(0) * h + beta.unsqueeze(0)

class SubgraphHGTLayer(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, n_heads, num_ntypes, dropout, context_dim):
        super().__init__()
        num_etypes = 1
        self.hgt_conv = dgl.nn.pytorch.conv.HGTConv(
            in_dim, h_dim, n_heads, num_ntypes, num_etypes, dropout, use_norm=True
        )
        if in_dim != out_dim:
            self.skip_proj = nn.Linear(in_dim, out_dim)
        else:
            self.skip_proj = nn.Identity()
        self.film_layer = FiLMLayer(out_dim, context_dim)

    def forward(self, subgraph, features_dict, global_context):
        if subgraph.num_edges() == 0:
            modulated_features = {}
            for ntype in subgraph.ntypes:
                if ntype in features_dict:
                    feat = features_dict[ntype]
                    skipped_feat = self.skip_proj(feat)
                    modulated_features[ntype] = self.film_layer(skipped_feat, global_context)
            return modulated_features

        skip_features = {ntype: self.skip_proj(features_dict[ntype]) for ntype in subgraph.ntypes}
        
        for ntype, feat in features_dict.items():
            if ntype in subgraph.ntypes:
                subgraph.nodes[ntype].data['h'] = feat
        
        homo_g = dgl.to_homogeneous(subgraph, ndata=['h'])
        h, ntype, etype = homo_g.ndata['h'], homo_g.ndata[dgl.NTYPE], homo_g.edata[dgl.ETYPE]
        
        hgt_homo_out = self.hgt_conv(homo_g, h, ntype, etype, presorted=False)
        
        hgt_out_dict = {}
        offset = 0
        for ntype_name in subgraph.ntypes:
            num_nodes = subgraph.num_nodes(ntype_name)
            hgt_out_dict[ntype_name] = hgt_homo_out[offset : offset + num_nodes]
            offset += num_nodes
        
        final_out_dict = {}
        for ntype, h_vec in hgt_out_dict.items():
            h_res = h_vec + skip_features[ntype]
            h_calibrated = self.film_layer(h_res, global_context)
            final_out_dict[ntype] = h_calibrated
            
        return final_out_dict

class AMNTDDA(nn.Module):
    def __init__(self, args, device=_device):
        super(AMNTDDA, self).__init__()
        self.args = args
        self.device = device       
        self.gt_drug = gt_drug.GraphTransformer(self.device, args.gt_layer, args.drug_gt_original_feature_dim, args.gt_out_dim, args.gt_out_dim, args.gt_head, args.dropout).to(self.device)
        self.gt_disease = gt_disease.GraphTransformer(self.device, args.gt_layer, args.disease_gt_original_feature_dim, args.gt_out_dim, args.gt_out_dim, args.gt_head, args.dropout).to(self.device)
        self.drug_linear_hgt = nn.Linear(args.drug_feature_raw_dim, args.hgt_in_dim).to(self.device)
        self.disease_linear_hgt = nn.Linear(args.disease_feature_raw_dim, args.hgt_in_dim).to(self.device)
        self.protein_linear_hgt = nn.Linear(args.protein_feature_raw_dim, args.hgt_in_dim).to(self.device)      
        hgt_out_dim = args.hgt_head_dim * args.hgt_head     
        from dgl.nn.pytorch.glob import GlobalAttentionPooling
        self.context_dim = 64 
        num_context_heads = 4
        if self.context_dim % num_context_heads != 0:
            raise ValueError("context_dim must be divisible by num_context_heads")
        context_gnn_out_dim_per_head = self.context_dim // num_context_heads 
        self.global_context_gnn = dgl.nn.pytorch.GATConv(
            args.hgt_in_dim, 
            context_gnn_out_dim_per_head, 
            num_heads=num_context_heads, 
            allow_zero_in_degree=True
        ).to(device)
        self.global_context_readout = GlobalAttentionPooling(
            gate_nn=nn.Linear(self.context_dim, 1)
        ).to(device)
        self.hgt_layer_A = SubgraphHGTLayer(args.hgt_in_dim, args.hgt_head_dim, hgt_out_dim, args.hgt_head, 2, args.dropout, self.context_dim).to(device)
        self.hgt_layer_B = SubgraphHGTLayer(args.hgt_in_dim, args.hgt_head_dim, hgt_out_dim, args.hgt_head, 2, args.dropout, self.context_dim).to(device)
        self.hgt_layer_C = SubgraphHGTLayer(args.hgt_in_dim, args.hgt_head_dim, hgt_out_dim, args.hgt_head, 2, args.dropout, self.context_dim).to(device)
        self.hgt_layer_D = SubgraphHGTLayer(args.hgt_in_dim, args.hgt_head_dim, hgt_out_dim, args.hgt_head, 2, args.dropout, self.context_dim).to(device)
        self.view_aggregator_drug = ViewAttention(hgt_out_dim)
        self.view_aggregator_disease = ViewAttention(hgt_out_dim)       
        if hgt_out_dim != args.gt_out_dim:
            self.project_hgt_drug_to_fusion = nn.Linear(hgt_out_dim, args.gt_out_dim).to(self.device)
            self.project_hgt_disease_to_fusion = nn.Linear(hgt_out_dim, args.gt_out_dim).to(self.device)
        else:
            self.project_hgt_drug_to_fusion = nn.Identity()
            self.project_hgt_disease_to_fusion = nn.Identity()
        encoder_layer = nn.TransformerEncoderLayer(d_model=args.gt_out_dim, nhead=args.tr_head, batch_first=True, dropout=args.dropout).to(self.device)
        self.drug_trans_fusion = nn.TransformerEncoder(encoder_layer, num_layers=args.tr_layer).to(self.device)
        self.disease_trans_fusion = nn.TransformerEncoder(encoder_layer, num_layers=args.tr_layer).to(self.device)       
        self.mlp = nn.Sequential(
                nn.Linear(args.gt_out_dim, 1024), nn.ReLU(), nn.Dropout(0.4),
                nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.4),
                nn.Linear(256, 2)
            ).to(self.device)            
        self.gt_drug_perturb = None
        self.gt_disease_perturb = None
        self.hgt_input_features_perturb = None

    def forward(self, drdr_graph, didi_graph, base_hg, metapath_subgraphs,
                drug_feature_for_hgt_raw, disease_feature_for_hgt_raw, protein_feature_for_hgt_raw,
                sample, return_intermediate_features=False, use_provided_features=False,
                provided_gt_drug_h_intermediate=None, provided_gt_disease_h_intermediate=None, provided_hgt_input_features=None):
        
        self.gt_drug.node_perturb = self.gt_drug_perturb
        self.gt_disease.node_perturb = self.gt_disease_perturb
        dr_sim = self.gt_drug(drdr_graph, h_intermediate=provided_gt_drug_h_intermediate if use_provided_features else None)
        di_sim = self.gt_disease(didi_graph, h_intermediate=provided_gt_disease_h_intermediate if use_provided_features else None)        
        if use_provided_features and provided_hgt_input_features is not None:
             drug_feat_proj, disease_feat_proj, protein_feat_proj = torch.split(provided_hgt_input_features, [self.args.drug_number, self.args.disease_number, self.args.protein_number])
        else:
             drug_feat_proj = self.drug_linear_hgt(drug_feature_for_hgt_raw.to(self.device))
             disease_feat_proj = self.disease_linear_hgt(disease_feature_for_hgt_raw.to(self.device))
             protein_feat_proj = self.protein_linear_hgt(protein_feature_for_hgt_raw.to(self.device))
        if self.hgt_input_features_perturb is not None and not use_provided_features:
            p_dr, p_di, p_pr = torch.split(self.hgt_input_features_perturb.to(self.device), [self.args.drug_number, self.args.disease_number, self.args.protein_number])
            drug_feat_proj, disease_feat_proj, protein_feat_proj = drug_feat_proj + p_dr, disease_feat_proj + p_di, protein_feat_proj + p_pr       
        base_hg = base_hg.to(self.device)
        if base_hg.num_nodes() > 0:
            initial_hgt_features_dict = {
                'drug': drug_feat_proj, 'disease': disease_feat_proj, 'protein': protein_feat_proj
            }
            for ntype, feat in initial_hgt_features_dict.items():
                if ntype in base_hg.ntypes:
                    base_hg.nodes[ntype].data['h'] = feat            
            homo_base_hg = dgl.to_homogeneous(base_hg, ndata=['h'])
            context_features = self.global_context_gnn(homo_base_hg, homo_base_hg.ndata['h']).flatten(1)
            global_context_vector = self.global_context_readout(homo_base_hg, context_features).squeeze(0)
        else:
            global_context_vector = torch.zeros(self.context_dim, device=self.device)
        g_a = metapath_subgraphs['A_d_p_di'].to(self.device)
        features_for_a = {'drug': drug_feat_proj, 'disease': disease_feat_proj}
        view_a_out = self.hgt_layer_A(g_a, features_for_a, global_context_vector)
        h_dr_view_a = view_a_out.get('drug')
        h_di_view_a = view_a_out.get('disease')        
        g_b = metapath_subgraphs['B_di_p_d'].to(self.device)
        features_for_b = {'drug': drug_feat_proj, 'disease': disease_feat_proj}
        view_b_out = self.hgt_layer_B(g_b, features_for_b, global_context_vector)
        h_dr_view_b = view_b_out.get('drug')
        h_di_view_b = view_b_out.get('disease')
        g_c = metapath_subgraphs['C_di_d_p'].to(self.device)
        features_for_c = {'disease': disease_feat_proj, 'protein': protein_feat_proj}
        view_c_out = self.hgt_layer_C(g_c, features_for_c, global_context_vector)
        h_di_view_c = view_c_out.get('disease')       
        g_d = metapath_subgraphs['D_d_di_p'].to(self.device)
        features_for_d = {'drug': drug_feat_proj, 'protein': protein_feat_proj}
        view_d_out = self.hgt_layer_D(g_d, features_for_d, global_context_vector)
        h_dr_view_d = view_d_out.get('drug')        
        dr_views = [h_dr_view_a, h_dr_view_b, h_dr_view_d]
        di_views = [h_di_view_a, h_di_view_b, h_di_view_c]        
        dr_hgt = self.view_aggregator_drug(dr_views)
        if dr_hgt is None:
            dr_hgt = torch.zeros(self.args.drug_number, hgt_out_dim).to(self.device)           
        di_hgt = self.view_aggregator_disease(di_views)
        if di_hgt is None:
            di_hgt = torch.zeros(self.args.disease_number, hgt_out_dim).to(self.device)
        dr_hgt_projected = self.project_hgt_drug_to_fusion(dr_hgt)
        di_hgt_projected = self.project_hgt_disease_to_fusion(di_hgt)       
        dr_stacked = torch.stack((dr_sim, dr_hgt_projected), dim=1)
        di_stacked = torch.stack((di_sim, di_hgt_projected), dim=1)       
        dr_fused_seq = self.drug_trans_fusion(dr_stacked)
        di_fused_seq = self.disease_trans_fusion(di_stacked)      
        dr_final_rep = (dr_fused_seq[:, 0, :] + dr_fused_seq[:, 1, :]) / 2
        di_final_rep = (di_fused_seq[:, 0, :] + di_fused_seq[:, 1, :]) / 2      
        if sample is None:
            if return_intermediate_features:
                return dr_final_rep, di_final_rep, dr_sim, di_sim, dr_hgt, di_hgt, None
            else:
                return dr_final_rep, di_final_rep, None
        output_logits = torch.empty(0, 2, device=self.device) 
        if sample.numel() > 0:
            sample_device = sample.to(self.device)
            dr_indices = sample_device[:, 0]
            di_indices = sample_device[:, 1]
            selected_dr_rep = dr_final_rep[dr_indices]
            selected_di_rep = di_final_rep[di_indices]
            drdi_embedding = torch.mul(selected_dr_rep, selected_di_rep)
            output_logits = self.mlp(drdi_embedding)           
        if return_intermediate_features:
             return dr_final_rep, di_final_rep, dr_sim, di_sim, dr_hgt, di_hgt, output_logits
        else:

             return dr_final_rep, di_final_rep, output_logits



