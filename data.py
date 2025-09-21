import numpy as np
import random
import torch
import pandas as pd
import dgl
import networkx as nx
from sklearn.model_selection import StratifiedKFold
from scipy.sparse import csr_matrix
import warnings

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def get_adj(edges, size):
    edges_tensor = torch.LongTensor(edges).t()
    values = torch.ones(len(edges))
    adj = torch.sparse.LongTensor(edges_tensor, values, size).to_dense().long()
    return adj


def k_matrix(matrix, k):
    num = matrix.shape[0]
    knn_graph = np.zeros(matrix.shape)
    idx_sort = np.argsort(-(matrix - np.eye(num)), axis=1)
    for i in range(num):
        knn_graph[i, idx_sort[i, :k + 1]] = matrix[i, idx_sort[i, :k + 1]]
        knn_graph[idx_sort[i, :k + 1], i] = matrix[idx_sort[i, :k + 1], i]
    return knn_graph + np.eye(num)


def get_data(args):
    data = dict()

    drf = pd.read_csv(args.data_dir + 'DrugFingerprint.csv').iloc[:, 1:].to_numpy()
    drg = pd.read_csv(args.data_dir + 'DrugGIP.csv').iloc[:, 1:].to_numpy()

    dip = pd.read_csv(args.data_dir + 'DiseasePS.csv').iloc[:, 1:].to_numpy()
    dig = pd.read_csv(args.data_dir + 'DiseaseGIP.csv').iloc[:, 1:].to_numpy()

    data['drug_number'] = int(drf.shape[0])
    data['disease_number'] = int(dig.shape[0])

    data['drf'] = drf
    data['drg'] = drg
    data['dip'] = dip
    data['dig'] = dig

    data['drdi'] = pd.read_csv(args.data_dir + 'DrugDiseaseAssociationNumber.csv', dtype=int).to_numpy()
    data['drpr'] = pd.read_csv(args.data_dir + 'DrugProteinAssociationNumber.csv', dtype=int).to_numpy()
    # 注意文件名可能是 ProteinDisease...
    data['dipr'] = pd.read_csv(args.data_dir + 'ProteinDiseaseAssociationNumber.csv', dtype=int).to_numpy() 

    data['drugfeature'] = pd.read_csv(args.data_dir + 'Drug_mol2vec.csv', header=None).iloc[:, 1:].to_numpy()
    data['diseasefeature'] = pd.read_csv(args.data_dir + 'DiseaseFeature.csv', header=None).iloc[:, 1:].to_numpy()
    data['proteinfeature'] = pd.read_csv(args.data_dir + 'Protein_ESM.csv', header=None).iloc[:, 1:].to_numpy()
    data['protein_number']= data['proteinfeature'].shape[0]

    return data


def data_processing(data, args):
    """
    数据处理总函数。
    【修改】: 不再进行负采样。只分离正负样本池，并将它们传递给主训练循环。
    负采样将在每一轮训练中动态进行。
    """
    drdi_matrix = get_adj(data['drdi'], (args.drug_number, args.disease_number))
    one_index = []
    zero_index = []
    for i in range(drdi_matrix.shape[0]):
        for j in range(drdi_matrix.shape[1]):
            if drdi_matrix[i][j] >= 1:
                one_index.append([i, j])
            else:
                zero_index.append([i, j])
                
    random.seed(args.random_seed)
    random.shuffle(one_index)
    
    print("Data processing complete. Positive and negative pools are ready for dynamic sampling.")

    # 【新增】: 将正负样本池直接存入data字典
    data['positive_samples'] = np.array(one_index, dtype=int)
    data['negative_pool'] = np.array(zero_index, dtype=int)
    
    drs_mean = (data['drf'] + data['drg']) / 2
    dis_mean = (data['dip'] + data['dig']) / 2
    data['drs'] = np.where(data['drf'] == 0, data['drg'], drs_mean)
    data['dis'] = np.where(data['dip'] == 0, data['dig'], dis_mean)
    
    return data


def k_fold(data, args, fold_num, current_epoch_sampled_negatives):
    k = args.k_fold
    one_index = data['positive_samples']
    zero_index = current_epoch_sampled_negatives
    index = np.array(list(one_index) + list(zero_index), dtype=int)
    label = np.array([1] * len(one_index) + [0] * len(zero_index), dtype=int)
    skf = StratifiedKFold(n_splits=k, random_state=args.random_seed, shuffle=True)
    all_splits = list(skf.split(index, label))
    train_index, test_index = all_splits[fold_num]
    X_train, X_test = index[train_index], index[test_index]
    Y_train, Y_test = label[train_index], label[test_index]
    Y_train = np.expand_dims(Y_train, axis=1).astype('float64')
    Y_test = np.expand_dims(Y_test, axis=1).astype('float64')
    fold_data = {
        'X_train': X_train,
        'X_test': X_test,
        'Y_train': Y_train,
        'Y_test': Y_test
    }
    return fold_data


def dgl_similarity_graph(data, args):
    drdr_matrix = k_matrix(data['drs'], args.neighbor)
    didi_matrix = k_matrix(data['dis'], args.neighbor)

    drdr_nx = nx.from_numpy_array(drdr_matrix)
    didi_nx = nx.from_numpy_array(didi_matrix)

    drdr_graph = dgl.from_networkx(drdr_nx)
    didi_graph = dgl.from_networkx(didi_nx)

    drdr_graph.ndata['drs'] = torch.tensor(data['drs'])
    didi_graph.ndata['dis'] = torch.tensor(data['dis'])

    return drdr_graph, didi_graph, data


# --- 【MODIFIED】: 修改以支持双向边 ---
def dgl_heterograph(data, drdi_train_pairs, args):
    drdi_list = [pair for pair in drdi_train_pairs]
    drpr_list = [pair for pair in data['drpr']]
    dipr_list = [pair for pair in data['dipr']] 

    node_dict = {
        'drug': args.drug_number,
        'disease': args.disease_number,
        'protein': args.protein_number
    }

    # --- 【MODIFIED】: 定义双向边类型 ---
    heterograph_dict = {
        ('drug', 'ddi', 'disease'): drdi_list,
        ('disease', 'did', 'drug'): [(j, i) for i, j in drdi_list], # 逆关系
        
        ('drug', 'dpr', 'protein'): drpr_list,
        ('protein', 'prd', 'drug'): [(j, i) for i, j in drpr_list], # 逆关系
        
        ('disease', 'dip', 'protein'): dipr_list,
        ('protein', 'pid', 'disease'): [(j, i) for i, j in dipr_list]  # 逆关系
    }

    # 移除没有边的关系，防止DGL报错
    final_heterograph_dict = {k: v for k, v in heterograph_dict.items() if v}

    base_hg = dgl.heterograph(final_heterograph_dict, num_nodes_dict=node_dict)

    data['feature_dict'] ={
        'drug': torch.tensor(data['drugfeature']),
        'disease': torch.tensor(data['diseasefeature']),
        'protein': torch.tensor(data['proteinfeature'])
    }

    return base_hg, data

# --- 【NEW】: 新增函数，用于生成元路径子图 ---
def generate_metapath_subgraphs(base_hg):
    """
    根据预定义的四个元路径，从基础异构图中抽取出子图。
    """
    metapaths = {
        # 预测路径
        'A_d_p_di': [('dpr'), ('pid')], # 药物 -> 蛋白质 -> 疾病
        'B_di_p_d': [('dip'), ('prd')], # 疾病 -> 蛋白质 -> 药物
        # 特征增强路径
        'C_di_d_p': [('did'), ('dpr')], # 疾病 -> 药物 -> 蛋白质
        'D_d_di_p': [('ddi'), ('dip')]  # 药物 -> 疾病 -> 蛋白质
    }
    
    subgraphs = {}
    print("Generating metapath subgraphs...")
    for name, path in metapaths.items():
        # dgl.metapath_reachable_graph 会返回一个只包含起点和终点类型节点的新图
        # 图中的边表示由该元路径连接的节点对
        try:
            subgraphs[name] = dgl.metapath_reachable_graph(base_hg, path)
            print(f"  - Subgraph '{name}' created with {subgraphs[name].num_nodes()} nodes and {subgraphs[name].num_edges()} edges.")
        except KeyError as e:
            print(f"  - Warning: Could not create subgraph '{name}'. Metapath edge type {e} not found in base graph. Graph will be empty.")
            subgraphs[name] = dgl.heterograph({(path[0][0], f'meta_{name}', path[-1][-1]): []},
                                               num_nodes_dict={ntype: base_hg.num_nodes(ntype) for ntype in base_hg.ntypes})

    return subgraphs

