import timeit
import argparse
import random
import os
import numpy as np
import pandas as pd
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import get_data, data_processing, k_fold, dgl_similarity_graph, dgl_heterograph, generate_metapath_subgraphs
from model import AMNTDDA
from metric import get_metric
from sklearn.metrics import roc_curve, precision_recall_curve

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def dynamic_negative_sampling_disentangled(
    model, all_drug_rep, all_disease_rep, positive_samples, negative_pool, args,
    is_test=False, num_samples_to_mine=None
):
  
    model.eval()
    with torch.no_grad():
        k_rank = getattr(args, 'k_rank', 32)
        hard_pool_ratio = getattr(args, 'hard_pool_ratio', 0.3)
        easy_pool_ratio = getattr(args, 'easy_pool_ratio', 0.5)

        def disentangle(features):
            if features.shape[0] == 0 or features.shape[1] < k_rank:
                return torch.zeros_like(features), torch.zeros_like(features)
            try:
                U, S, V = torch.svd(features)
            except torch.linalg.LinAlgError:
                return features, torch.zeros_like(features)
            low_rank_S = torch.diag(S[:k_rank])
            low_rank_features = U[:, :k_rank] @ low_rank_S @ V[:, :k_rank].T
            wider_rank_features = features - low_rank_features
            return low_rank_features, wider_rank_features

        drug_rep_low, drug_rep_wider = disentangle(all_drug_rep)
        disease_rep_low, disease_rep_wider = disentangle(all_disease_rep)

        drug_low_norm = F.normalize(drug_rep_low, p=2, dim=1)
        disease_low_norm = F.normalize(disease_rep_low, p=2, dim=1)
        low_rank_scores_tensor = drug_low_norm @ disease_low_norm.T
        drug_wider_norm = F.normalize(drug_rep_wider, p=2, dim=1)
        disease_wider_norm = F.normalize(disease_rep_wider, p=2, dim=1)
        wider_rank_scores_tensor = drug_wider_norm @ disease_wider_norm.T
        rectification_factor = 1.0 + torch.tanh(wider_rank_scores_tensor)
        combined_scores_tensor = low_rank_scores_tensor * rectification_factor

        combined_scores = combined_scores_tensor.cpu().numpy()
        drug_indices = negative_pool[:, 0]
        disease_indices = negative_pool[:, 1]
        hardness_scores = combined_scores[drug_indices, disease_indices]

        sorted_indices = np.argsort(-hardness_scores)
        sorted_negatives = negative_pool[sorted_indices]

        if is_test:
            if num_samples_to_mine is None or num_samples_to_mine <= 0:
                return np.array([])
            return sorted_negatives[:num_samples_to_mine]
        else:
            total_neg_pool_size = len(sorted_negatives)
            hard_pool_end_idx = int(total_neg_pool_size * hard_pool_ratio)
            easy_pool_start_idx = int(total_neg_pool_size * (1 - easy_pool_ratio))
            hard_pool = sorted_negatives[:hard_pool_end_idx]
            easy_pool = sorted_negatives[easy_pool_start_idx:]

            num_negatives_to_sample = int(args.negative_rate * len(positive_samples))
            if num_negatives_to_sample == 0: return np.array([])

            hard_ratio_for_sampling = getattr(args, 'hard_ratio', 0.8)
            num_hard_samples = int(num_negatives_to_sample * hard_ratio_for_sampling)
            num_easy_samples = num_negatives_to_sample - num_hard_samples

            if len(hard_pool) > 0:
                hard_indices = np.random.choice(len(hard_pool), min(num_hard_samples, len(hard_pool)), replace=False)
                sampled_hard = hard_pool[hard_indices]
            else:
                sampled_hard = np.array([])

            if len(easy_pool) > 0 and num_easy_samples > 0:
                easy_indices = np.random.choice(len(easy_pool), min(num_easy_samples, len(easy_pool)), replace=False)
                sampled_easy = easy_pool[easy_indices]
            else:
                sampled_easy = np.array([])

            if len(sampled_hard) > 0 and len(sampled_easy) > 0:
                sampled_negatives = np.vstack((sampled_hard, sampled_easy))
            elif len(sampled_hard) > 0:
                sampled_negatives = sampled_hard
            elif len(sampled_easy) > 0:
                sampled_negatives = sampled_easy
            else:
                sampled_negatives = sorted_negatives[:num_negatives_to_sample]

            if len(sampled_negatives) > 0:
                np.random.shuffle(sampled_negatives)

            model.train()
            return sampled_negatives

def calculate_agas_loss(h_orig, h_adv, lambda_align, lambda_sep, m_align, margin_sep, gamma_sep, tau_sep=0.1, distance_type='cosine'):
    N = h_orig.shape[0]
    if N <= 1:
        return torch.tensor(0.0, device=h_orig.device), torch.tensor(0.0, device=h_orig.device), torch.tensor(0.0, device=h_orig.device)
    h_orig_norm = F.normalize(h_orig, p=2, dim=1)
    h_adv_norm = F.normalize(h_adv, p=2, dim=1)
    if distance_type == 'cosine':
        distance = 1 - F.cosine_similarity(h_orig_norm, h_adv_norm, dim=-1)
    else:
        distance = torch.norm(h_orig_norm - h_adv_norm, p=2, dim=1)
    perturbation_distance = distance.detach()
    alignment_loss = F.relu(distance - m_align).mean()
    sim_adv_orig_neg = torch.matmul(h_adv_norm, h_orig_norm.T) / tau_sep
    mask_neg = (~torch.eye(N, dtype=torch.bool, device=device)).float()
    sim_adv_orig_neg = sim_adv_orig_neg * mask_neg
    separation_loss_matrix = F.relu(sim_adv_orig_neg - margin_sep)
    separation_weights = (1 + gamma_sep * perturbation_distance).unsqueeze(1)
    weighted_separation_loss = (separation_weights * separation_loss_matrix * mask_neg).sum(dim=1) / (N - 1 + 1e-9)
    separation_loss = weighted_separation_loss.mean()
    total_agas_loss = lambda_align * alignment_loss + lambda_sep * separation_loss
    return total_agas_loss, alignment_loss, separation_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--k_fold', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--k_rank', type=int, default=32, help="Dimension of the low-rank space for disentanglement.")
    parser.add_argument('--score_weight', type=float, default=0.5, help="Weight for combining overall score and low-rank score. w * overall + (1-w) * low_rank")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--random_seed', type=int, default=1234)
    parser.add_argument('--neighbor', type=int, default=20)
    parser.add_argument('--negative_rate', type=float, default=1.0)
    parser.add_argument('--hard_ratio', type=float, default=0.8, help="Ratio of hard negatives in dynamic sampling")
    parser.add_argument('--dataset', default='F-dataset')
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--gt_layer', default=2, type=int)
    parser.add_argument('--gt_head', default=4, type=int)
    parser.add_argument('--gt_out_dim', default=200, type=int)
    parser.add_argument('--drug_gt_original_feature_dim', type=int, default=663)
    parser.add_argument('--disease_gt_original_feature_dim', type=int, default=409)
    parser.add_argument('--hgt_layer', default=2, type=int)
    parser.add_argument('--hgt_head', default=8, type=int)
    parser.add_argument('--hgt_in_dim', default=64, type=int)
    parser.add_argument('--hgt_head_dim', default=25, type=int)
    parser.add_argument('--num_hgt_canonical_etypes', type=int, default=6) 
    parser.add_argument('--drug_feature_raw_dim', type=int, default=300)
    parser.add_argument('--disease_feature_raw_dim', type=int, default=64)
    parser.add_argument('--protein_feature_raw_dim', type=int, default=320)
    parser.add_argument('--tr_layer', default=2, type=int)
    parser.add_argument('--tr_head', default=4, type=int)
    parser.add_argument('--adv_epsilon_gt', type=float, default=0.01)
    parser.add_argument('--adv_epsilon_hgt', type=float, default=0.01)
    parser.add_argument('--alpha_adv', type=float, default=0.1)
    parser.add_argument('--agas_lambda_align', type=float, default=0.5)
    parser.add_argument('--agas_lambda_sep', type=float, default=0.5)
    parser.add_argument('--agas_m_align', type=float, default=0.1)
    parser.add_argument('--agas_margin_sep', type=float, default=0.5)
    parser.add_argument('--agas_gamma_sep', type=float, default=0.1)
    parser.add_argument('--agas_tau_sep', type=float, default=0.1)
    parser.add_argument('--agas_distance_type', type=str, default='cosine', choices=['cosine', 'l2'])

    args = parser.parse_args()
    args.data_dir = f'data/{args.dataset}/'
    agas_suffix = f"_AGAS_MetaPathHGT_v2" 
    args.result_dir = f"results/{args.dataset}/AMNTDDA{agas_suffix}/"
    os.makedirs(args.result_dir, exist_ok=True)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.random_seed)
    random.seed(args.random_seed)
    all_data_dict = get_data(args)
    args.drug_number = all_data_dict['drug_number']
    args.disease_number = all_data_dict['disease_number']
    args.protein_number = all_data_dict['protein_number']
    args.drug_gt_original_feature_dim = args.drug_number
    args.disease_gt_original_feature_dim = args.disease_number
    all_data_dict = data_processing(all_data_dict, args)
    drdr_graph, didi_graph, all_data_dict = dgl_similarity_graph(all_data_dict, args)
    drdr_graph = drdr_graph.to(device)
    didi_graph = didi_graph.to(device)
    drug_feature_hgt_raw_tensor = torch.FloatTensor(all_data_dict['drugfeature']).to(device)
    disease_feature_hgt_raw_tensor = torch.FloatTensor(all_data_dict['diseasefeature']).to(device)
    protein_feature_hgt_raw_tensor = torch.FloatTensor(all_data_dict['proteinfeature']).to(device)
    start_time_total = timeit.default_timer()
    cross_entropy = nn.CrossEntropyLoss()
    AUCs, AUPRs = [], []
    Metric_Header = ('Fold\tEpoch\tTime\tL(T)\tL(A)\tL(AGAS)\tL(Algn)\tL(Sep)\tAUC\tAUPR\tAcc\tPrec\tRecall\tF1\tMCC')
    for i_fold in range(args.k_fold):
        print(f"\n---Fold: {i_fold}---")
        model = AMNTDDA(args, device=device)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
        print(Metric_Header)
        best_auc_fold, best_aupr_fold = 0.0, 0.0
        best_epoch_fold = -1
        fold_start_time = timeit.default_timer()
        num_pos = len(all_data_dict['positive_samples'])
        temp_negatives_for_split = all_data_dict['negative_pool'][:num_pos]
        fold_split_info = k_fold(all_data_dict, args, i_fold, temp_negatives_for_split)
        X_train_indices = fold_split_info['X_train']
        Y_train_labels = fold_split_info['Y_train'].flatten()
        training_positives_for_graph = X_train_indices[Y_train_labels == 1]
        base_hg, _ = dgl_heterograph(all_data_dict, training_positives_for_graph, args)
        args.num_hgt_canonical_etypes = len(base_hg.canonical_etypes) if base_hg.num_edges() > 0 else 1
        metapath_subgraphs = generate_metapath_subgraphs(base_hg)
        for epoch in range(args.epochs):
            model.eval()
            with torch.no_grad():
                all_drug_rep, all_disease_rep, _ = model(
                    drdr_graph, didi_graph, base_hg, metapath_subgraphs,
                    drug_feature_hgt_raw_tensor, disease_feature_hgt_raw_tensor, protein_feature_hgt_raw_tensor,
                    sample=None)
            if epoch == 0:
                num_neg_to_sample = int(args.negative_rate * len(all_data_dict['positive_samples']))
                neg_pool_indices = np.random.choice(len(all_data_dict['negative_pool']), num_neg_to_sample, replace=False)
                current_sampled_negatives = all_data_dict['negative_pool'][neg_pool_indices]
            else:
                current_sampled_negatives = dynamic_negative_sampling_disentangled(
                    model, all_drug_rep, all_disease_rep,
                    all_data_dict['positive_samples'], all_data_dict['negative_pool'], args)
            fold_data = k_fold(all_data_dict, args, i_fold, current_sampled_negatives)
            X_train = torch.LongTensor(fold_data['X_train']).to(device)
            Y_train = torch.LongTensor(fold_data['Y_train']).flatten().to(device)
            X_test = torch.LongTensor(fold_data['X_test']).to(device)
            Y_test_labels_np = torch.LongTensor(fold_data['Y_test']).flatten().cpu().numpy()    
            model.train()
            optimizer.zero_grad()
            with torch.no_grad():
                h_drug_gt_intermediate_orig = model.gt_drug.get_intermediate_h(drdr_graph)
                h_disease_gt_intermediate_orig = model.gt_disease.get_intermediate_h(didi_graph)
                drug_feat_proj_hgt = model.drug_linear_hgt(drug_feature_hgt_raw_tensor)
                disease_feat_proj_hgt = model.disease_linear_hgt(disease_feature_hgt_raw_tensor)
                protein_feat_proj_hgt = model.protein_linear_hgt(protein_feature_hgt_raw_tensor)
                hgt_input_features_orig = torch.cat((drug_feat_proj_hgt, disease_feat_proj_hgt, protein_feat_proj_hgt), dim=0)
            h_drug_gt_intermediate_orig_rg = h_drug_gt_intermediate_orig.clone().detach().requires_grad_(True)
            h_disease_gt_intermediate_orig_rg = h_disease_gt_intermediate_orig.clone().detach().requires_grad_(True)
            hgt_input_features_orig_rg = hgt_input_features_orig.clone().detach().requires_grad_(True)
            model.zero_grad()
            _, _, _, _, _, _, temp_train_score = model(
                drdr_graph, didi_graph, base_hg, metapath_subgraphs,
                drug_feature_hgt_raw_tensor, disease_feature_hgt_raw_tensor, protein_feature_hgt_raw_tensor,
                X_train, return_intermediate_features=True, use_provided_features=True,
                provided_gt_drug_h_intermediate=h_drug_gt_intermediate_orig_rg,
                provided_gt_disease_h_intermediate=h_disease_gt_intermediate_orig_rg,
                provided_hgt_input_features=hgt_input_features_orig_rg)
            if temp_train_score.numel() > 0 :
                loss_for_grad = cross_entropy(temp_train_score, Y_train)
                loss_for_grad.backward()
            perturbation_drug_gt = args.adv_epsilon_gt * h_drug_gt_intermediate_orig_rg.grad.data.sign()
            perturbation_disease_gt = args.adv_epsilon_gt * h_disease_gt_intermediate_orig_rg.grad.data.sign()
            perturbation_hgt_input = args.adv_epsilon_hgt * hgt_input_features_orig_rg.grad.data.sign()
            perturbation_drug_gt = perturbation_drug_gt.detach()
            perturbation_disease_gt = perturbation_disease_gt.detach()
            perturbation_hgt_input = perturbation_hgt_input.detach()
            optimizer.zero_grad()
            _, _, dr_sim_orig, di_sim_orig, dr_hgt_orig, di_hgt_orig, train_score_orig = model(
                drdr_graph, didi_graph, base_hg, metapath_subgraphs, drug_feature_hgt_raw_tensor,
                disease_feature_hgt_raw_tensor, protein_feature_hgt_raw_tensor, X_train, return_intermediate_features=True, use_provided_features=False)
            task_loss_orig = cross_entropy(train_score_orig, Y_train) if train_score_orig.numel() > 0 else torch.tensor(0.0, device=device)
            model.gt_drug_perturb = perturbation_drug_gt
            model.gt_disease_perturb = perturbation_disease_gt
            model.hgt_input_features_perturb = perturbation_hgt_input
            _, _, dr_sim_adv, di_sim_adv, dr_hgt_adv, di_hgt_adv, train_score_adv = model(
                drdr_graph, didi_graph, base_hg, metapath_subgraphs, drug_feature_hgt_raw_tensor,
                disease_feature_hgt_raw_tensor, protein_feature_hgt_raw_tensor, X_train, return_intermediate_features=True, use_provided_features=False)
            task_loss_adv = cross_entropy(train_score_adv, Y_train) if train_score_adv.numel() > 0 else torch.tensor(0.0, device=device)
            model.gt_drug_perturb = None; model.gt_disease_perturb = None; model.hgt_input_features_perturb = None
            agas_loss_dr_sim, align_dr_sim, sep_dr_sim = calculate_agas_loss(dr_sim_orig, dr_sim_adv, args.agas_lambda_align, args.agas_lambda_sep, args.agas_m_align, args.agas_margin_sep, args.agas_gamma_sep, args.agas_tau_sep, args.agas_distance_type)
            agas_loss_di_sim, align_di_sim, sep_di_sim = calculate_agas_loss(di_sim_orig, di_sim_adv, args.agas_lambda_align, args.agas_lambda_sep, args.agas_m_align, args.agas_margin_sep, args.agas_gamma_sep, args.agas_tau_sep, args.agas_distance_type)
            agas_loss_dr_hgt, align_dr_hgt, sep_dr_hgt = calculate_agas_loss(dr_hgt_orig, dr_hgt_adv, args.agas_lambda_align, args.agas_lambda_sep, args.agas_m_align, args.agas_margin_sep, args.agas_gamma_sep, args.agas_tau_sep, args.agas_distance_type)
            agas_loss_di_hgt, align_di_hgt, sep_di_hgt = calculate_agas_loss(di_hgt_orig, di_hgt_adv, args.agas_lambda_align, args.agas_lambda_sep, args.agas_m_align, args.agas_margin_sep, args.agas_gamma_sep, args.agas_tau_sep, args.agas_distance_type)
            total_agas_loss_value = (agas_loss_dr_sim + agas_loss_di_sim + agas_loss_dr_hgt + agas_loss_di_hgt) / 4.0
            align_loss_value = (align_dr_sim + align_di_sim + align_dr_hgt + align_di_hgt) / 4.0
            sep_loss_value = (sep_dr_sim + sep_di_sim + sep_dr_hgt + sep_di_hgt) / 4.0
            total_loss = task_loss_orig + 0.2 * task_loss_adv + total_agas_loss_value
            total_loss.backward()
            optimizer.step()
            epoch_task_loss = task_loss_orig.item()
            epoch_adv_loss = task_loss_adv.item()
            epoch_agas_loss = total_agas_loss_value.item()
            epoch_align_loss = align_loss_value.item()
            epoch_sep_loss = sep_loss_value.item()
            model.eval()
            with torch.no_grad():
                _, _, test_score_logits = model(
                    drdr_graph, didi_graph, base_hg, metapath_subgraphs, drug_feature_hgt_raw_tensor,
                    disease_feature_hgt_raw_tensor, protein_feature_hgt_raw_tensor, X_test, return_intermediate_features=False, use_provided_features=False)
            test_score_pred_np = torch.argmax(test_score_logits, dim=-1).cpu().numpy() if test_score_logits.numel() > 0 else np.array([])
            test_prob_np = F.softmax(test_score_logits, dim=-1)[:, 1].cpu().numpy() if test_score_logits.numel() > 0 else np.array([])
            AUC, AUPR, accuracy, precision_val, recall_val, f1, mcc = 0.0,0.0,0.0,0.0,0.0,0.0,0.0
            if Y_test_labels_np.size > 0 and test_score_pred_np.size > 0 and test_prob_np.size > 0 and len(np.unique(Y_test_labels_np)) > 1 :
                 AUC, AUPR, accuracy, precision_val, recall_val, f1, mcc = get_metric(Y_test_labels_np, test_score_pred_np, test_prob_np)
            current_loop_time = timeit.default_timer() - fold_start_time
            show_metrics = [ i_fold, epoch + 1, f"{current_loop_time:.2f}",
                             f"{epoch_task_loss:.4f}", f"{epoch_adv_loss:.4f}",
                             f"{epoch_agas_loss:.4f}", f"{epoch_align_loss:.4f}", f"{epoch_sep_loss:.4f}",
                             f"{AUC:.5f}", f"{AUPR:.5f}", f"{accuracy:.5f}",
                             f"{precision_val:.5f}", f"{recall_val:.5f}", f"{f1:.5f}", f"{mcc:.5f}" ]
            print('\t'.join(map(str, show_metrics)))
            if AUC > best_auc_fold and len(np.unique(Y_test_labels_np)) > 1:
                best_auc_fold = AUC
                best_aupr_fold = AUPR
                best_epoch_fold = epoch + 1
        AUCs.append(best_auc_fold)
        AUPRs.append(best_aupr_fold)
        print(f"--- Fold {i_fold} Finished --- Best AUC: {best_auc_fold:.5f} at Epoch {best_epoch_fold}, Best AUPR: {best_aupr_fold:.5f} ---")
    print("\n\n" + "="*50)
    print(" " * 12 + "FINAL K-FOLD CROSS-VALIDATION RESULTS")
    print("="*50)
    AUC_mean = np.mean(AUCs) if AUCs else 0.0
    AUC_std = np.std(AUCs) if AUCs else 0.0
    AUPR_mean = np.mean(AUPRs) if AUPRs else 0.0
    AUPR_std = np.std(AUPRs) if AUPRs else 0.0   
    print(f"\nDataset: {args.dataset}")
    print(f"Number of Folds: {args.k_fold}")   
    print("\n--- Performance Metrics ---")
    print(f"Mean AUC:  {AUC_mean:.5f} (± {AUC_std:.5f})")
    print(f"Mean AUPR: {AUPR_mean:.5f} (± {AUPR_std:.5f})")   
    print("\n--- AUC values for each fold ---")
    print(f"{AUCs}")    
    print("\n--- AUPR values for each fold ---")
    print(f"{AUPRs}")
    total_run_time = timeit.default_timer() - start_time_total
    print(f"\nTotal Execution Time: {total_run_time:.2f} seconds")
    print("="*50)



