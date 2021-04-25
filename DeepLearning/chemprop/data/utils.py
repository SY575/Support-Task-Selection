from argparse import Namespace
import csv
from logging import Logger
import pickle
import random
from typing import List, Set, Tuple
import os

from rdkit import Chem
import numpy as np
from tqdm import tqdm

from .data import MoleculeDatapoint, MoleculeDataset
from .scaffold import log_scaffold_stats, scaffold_split, scaffold_similarity_split
from chemprop.features import load_features


def get_task_names(path: str, use_compound_names: bool = False) -> List[str]:
    """
    Gets the task names from a data CSV file.

    :param path: Path to a CSV file.
    :param use_compound_names: Whether file has compound names in addition to smiles strings.
    :return: A list of task names.
    """
    index = 2 if use_compound_names else 1
    task_names = get_header(path)[index:]

    return task_names


def get_header(path: str) -> List[str]:
    """
    Returns the header of a data CSV file.

    :param path: Path to a CSV file.
    :return: A list of strings containing the strings in the comma-separated header.
    """
    with open(path) as f:
        header = next(csv.reader(f))

    return header


def get_num_tasks(path: str) -> int:
    """
    Gets the number of tasks in a data CSV file.

    :param path: Path to a CSV file.
    :return: The number of tasks.
    """
    return len(get_header(path)) - 1


def get_smiles(path: str, header: bool = True) -> List[str]:
    """
    Returns the smiles strings from a data CSV file (assuming the first line is a header).

    :param path: Path to a CSV file.
    :param header: Whether the CSV file contains a header (that will be skipped).
    :return: A list of smiles strings.
    """
    with open(path) as f:
        reader = csv.reader(f)
        if header:
            next(reader)  # Skip header
        smiles = [line[0] for line in reader]

    return smiles


def filter_invalid_smiles(data: MoleculeDataset) -> MoleculeDataset:
    """
    Filters out invalid SMILES.

    :param data: A MoleculeDataset.
    :return: A MoleculeDataset with only valid molecules.
    """
    return MoleculeDataset([datapoint for datapoint in data
                            if datapoint.smiles != '' and datapoint.mol is not None
                            and datapoint.mol.GetNumHeavyAtoms() > 0])


def get_data(path: str,
             skip_invalid_smiles: bool = True,
             args: Namespace = None,
             features_path: List[str] = None,
             max_data_size: int = None,
             use_compound_names: bool = None,
             logger: Logger = None) -> MoleculeDataset:
    """
    Gets smiles string and target values (and optionally compound names if provided) from a CSV file.

    :param path: Path to a CSV file.
    :param skip_invalid_smiles: Whether to skip and filter out invalid smiles.
    :param args: Arguments.
    :param features_path: A list of paths to files containing features. If provided, it is used
    in place of args.features_path.
    :param max_data_size: The maximum number of data points to load.
    :param use_compound_names: Whether file has compound names in addition to smiles strings.
    :param logger: Logger.
    :return: A MoleculeDataset containing smiles strings and target values along
    with other info such as additional features and compound names when desired.
    """
    debug = logger.debug if logger is not None else print

    if args is not None:
        # Prefer explicit function arguments but default to args if not provided
        features_path = features_path if features_path is not None else args.features_path
        max_data_size = max_data_size if max_data_size is not None else args.max_data_size
        use_compound_names = use_compound_names if use_compound_names is not None else args.use_compound_names
    else:
        use_compound_names = False

    max_data_size = max_data_size or float('inf')

    # Load features
    if features_path is not None:
        features_data = []
        for feat_path in features_path:
            features_data.append(load_features(feat_path))  # each is num_data x num_features
        features_data = np.concatenate(features_data, axis=1)
    else:
        features_data = None

    skip_smiles = set()

    # Load data
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header

        lines = []
        for line in reader:
            smiles = line[0]

            if smiles in skip_smiles:
                continue

            lines.append(line)

            if len(lines) >= max_data_size:
                break

        data = MoleculeDataset([
            MoleculeDatapoint(
                line=line,
                args=args,
                features=features_data[i] if features_data is not None else None,
                use_compound_names=use_compound_names
            ) for i, line in tqdm(enumerate(lines), total=len(lines))
        ])

    # Filter out invalid SMILES
    if skip_invalid_smiles:
        original_data_len = len(data)
        data = filter_invalid_smiles(data)

        if len(data) < original_data_len:
            debug(f'Warning: {original_data_len - len(data)} SMILES are invalid.')

    if data.data[0].features is not None:
        args.features_dim = len(data.data[0].features)

    return data


def get_data_from_smiles(smiles: List[str], skip_invalid_smiles: bool = True, logger: Logger = None) -> MoleculeDataset:
    """
    Converts SMILES to a MoleculeDataset.

    :param smiles: A list of SMILES strings.
    :param skip_invalid_smiles: Whether to skip and filter out invalid smiles.
    :param logger: Logger.
    :return: A MoleculeDataset with all of the provided SMILES.
    """
    debug = logger.debug if logger is not None else print

    data = MoleculeDataset([MoleculeDatapoint([smile]) for smile in smiles])

    # Filter out invalid SMILES
    if skip_invalid_smiles:
        original_data_len = len(data)
        data = filter_invalid_smiles(data)

        if len(data) < original_data_len:
            debug(f'Warning: {original_data_len - len(data)} SMILES are invalid.')

    return data


def split_data(data_raw: MoleculeDataset=None,
               split_type: str = 'random',
               sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
               seed: int = 0,
               args: Namespace = None,
               logger: Logger = None) -> Tuple[MoleculeDataset,
                                               MoleculeDataset,
                                               MoleculeDataset]:
    """
    Splits data into training, validation, and test splits.

    :param data: A MoleculeDataset.
    :param split_type: Split type.
    :param sizes: A length-3 tuple with the proportions of data in the
    train, validation, and test sets.
    :param seed: The random seed to use before shuffling data.
    :param args: Namespace of arguments.
    :param logger: A logger.
    :return: A tuple containing the train, validation, and test splits of the data.
    """
    assert len(sizes) == 3 and sum(sizes) == 1 and args is not None

# =============================================================================
#     if args is not None:
#         folds_file, val_fold_index, test_fold_index = \
#             args.folds_file, args.val_fold_index, args.test_fold_index
#     else:
#         folds_file = val_fold_index = test_fold_index = None
# =============================================================================
    
    if split_type == 'random':
        
        data_raw.shuffle(seed=seed)
        train_size = int(sizes[0] * len(data_raw))
        train_val_size = int((sizes[0] + sizes[1]) * len(data_raw))

        train_smiles = set([item.smiles for item in data_raw[:train_size]])
        val_smiles = set([item.smiles for item in data_raw[train_size:train_val_size]])
        test_smiles = set([item.smiles for item in data_raw[train_val_size:]])
        print('='*70)
        print(f'Raw dataset: train: {len(train_smiles)}, val: {len(val_smiles)}, test: {len(test_smiles)}')
        df, class_weights, task_type = task_selection(
                args, train_smiles, 
                ckpt='_'.join([key for key in args.datasets.keys()]))
        df.to_csv(args.data_path, index=False)
        args.class_weights = class_weights
        args.task_type = task_type
        args.minimize_score = True if task_type[0] == 'regression' else False
        args.num_tasks = len(class_weights)

        data = get_data(path=args.data_path, args=args)
        
        raw_smiles = train_smiles.union(val_smiles).union(test_smiles)
        idx_not_in_raw = np.array([idx for idx, item in enumerate(data) if item.smiles not in raw_smiles])
        random.shuffle(idx_not_in_raw)
        # train_size_r = int((sizes[0] + sizes[1]) * len(idx_not_in_raw))
        
        train_r = set(idx_not_in_raw)
        
        train_idx = np.array([idx for idx, item in enumerate(data) 
                              if item.smiles in train_smiles or idx in train_r])
        val_idx = np.array([idx for idx, item in enumerate(data) 
                            if item.smiles in val_smiles])
        test_idx = np.array([idx for idx, item in enumerate(data) 
                             if item.smiles in test_smiles])
        print(f'Train dataset: train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}')
        print('='*70)
        train = data[train_idx]
        val = data[val_idx]
        test = data[test_idx]
        
        return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test), args
    else:
        assert 0


import pandas as pd


from rdkit import Chem
from rdkit.Chem import AllChem
import scipy
from random import shuffle
from rdkit.Chem import MACCSkeys
p = './SMILES2VES.pkl'
if os.path.exists(p):
    SMILES2VES = pickle.load(open(p, 'rb'))
else:
    SMILES2VES = {}
    
def smiles2vec(smiles):
    if smiles in SMILES2VES:
        return SMILES2VES[smiles]
    else:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            emb = None
        else:
            emb = list(MACCSkeys.GenMACCSKeys(mol))
        SMILES2VES[smiles] = emb
        return SMILES2VES[smiles]

def task_selection(args, train_smiles, ckpt='all_chem2'):
    target_col = args.target_col
    # target_col = f'{target_col}_{args.dataset_type}'
    
    df_blank = pd.DataFrame({'smiles':[]})
    for dataset in args.datasets.keys():
        cols = pd.read_csv('../data/'+dataset+'.csv', nrows=1).columns.tolist()
        df0 = pd.read_csv('../data/'+dataset+'.csv', dtype={col:np.float16 for col in cols[1:]})
        df0.columns = ['smiles'] + [f'{dataset}_{col}' for col in df0.columns[1:].tolist()]
        df_blank = pd.merge(df_blank, df0, on='smiles', how='outer')
    del df0
    assert target_col in df_blank, f'Invalid target_col: {target_col}'
    
    train_smiles = list(train_smiles)
    if len(train_smiles) > 2000:
        shuffle(train_smiles)
        demo_smiles = train_smiles[:2000]
    else:
        demo_smiles = train_smiles
    X = np.array([smiles2vec(s) for s in demo_smiles])
# =============================================================================
#     X = np.array([AllChem.GetMorganFingerprintAsBitVect(
#             Chem.MolFromSmiles(s), 2, nBits=2048) for s in train_smiles])
# =============================================================================
    col_lst = ['smiles', target_col]
    class_weights = [1.]
    
    result = {}
    for dataset in args.datasets:
        cols = pd.read_csv(f'../data/{dataset}.csv', nrows=1).columns.tolist()
        cols = cols[1:]
        for label in cols:
            fp = f'../ckpt_STL/clf_{dataset}_{label}.pkl'
            try:
                clf = pickle.load(open(fp, 'rb'))
                result[f'{dataset}_{label}'] = clf.predict_proba(X)[:, 1]
            except:
                try:
                    clf = pickle.load(open(fp, 'rb'))
                    result[f'{dataset}_{label}'] = clf.predict(X)
                except:
                    print(fp)
    keys = sorted(list(result.keys()))
    keys = np.array([item for item in keys if item != args.target_col])
    pcc = np.array([abs(scipy.stats.spearmanr(
        result[target_col], result[keys[i]]).correlation) for i in range(len(keys))])
# =============================================================================
#     pcc = np.array([abs(np.corrcoef(result[target_col], 
#                                     result[keys[i]])[0][1]) 
#                     for i in range(len(keys))])
# =============================================================================
    T = 0.35
    col_lst += list(keys[np.where(pcc>T)])
    class_weights += list(pcc[np.where(pcc>T)])
    class_weights = np.array(class_weights)
    df_blank = df_blank[col_lst]
    
    # datapoint selection
    df_blank['tmp'] = df_blank[col_lst[1:]].fillna(-1).values.tolist()
    df_blank = df_blank.loc[df_blank['tmp'].apply(lambda x: sum(x))!=-(len(col_lst)-1)].reset_index(drop=True)
    
    df_blank['tmp'] = df_blank[col_lst[1:]].fillna(-1).values.tolist()
    df_blank = df_blank.loc[df_blank['tmp'].apply(
        lambda x: 1 if (x[0]!= -1 or 
                        np.sum(np.where(np.array(x)==-1, 0, 1)*np.where(class_weights>0.1, 10, class_weights))>1) else 0)==1].reset_index(drop=True)
    del df_blank['tmp']
    class_weights_new = [1.0]
    for i, col in enumerate(col_lst[2:]):
        if sum(~df_blank[col].isna()) < 100:
            del df_blank[col]
        else:
            class_weights_new.append(class_weights[i+1])
    class_weights = np.array(class_weights_new)
    col_lst = df_blank.columns.tolist()
    print(df_blank.shape)
    
    # 列名后处理
    # df_blank.columns = ['smiles'] + ['_'.join(item.split('_')[:-1]) for item in df_blank.columns[1:]]
    # target_col = '_'.join(target_col.split('_')[:-1])
    task_type = [args.datasets[dataset] for item in col_lst[1:]]
    
    return df_blank, class_weights, task_type
# =============================================================================
# def task_selection(args, test_smiles, ckpt='all_chem2'):
#     import torch
#     import gc
#     target_col = args.target_col
#     target_col = f'{target_col}_{args.dataset_type}'
#     device = torch.device('cpu')#torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
#     
#     df_blank = load_data(args, ckpt)
#     assert target_col in df_blank, f'Invalid target_col: {target_col}'
#     
#     df_blank = df_blank.loc[~df_blank.smiles.isin(test_smiles)].reset_index(drop=True)
#     gc.collect()
#     col_lst = ['smiles', target_col]
#     class_weights = [1.]
#     print(df_blank.shape)
#     df1 = df_blank[[target_col, 'emb']].loc[~df_blank[target_col].isna()]
#     mat1 = np.array(df1['emb'].tolist())
#     mat1 = torch.Tensor(mat1).to(device)
#     mat2 = np.array(df_blank['emb'].tolist()).transpose(1,0)
#     mat2 = torch.Tensor(mat2).to(device)
#     mat = mat1 @ mat2
#     task_cols = df_blank.columns[1:].tolist()
#     for col in tqdm(task_cols):
#         if col in [target_col, 'emb']:
#             continue
#         df2 = df_blank[[col]].loc[~df_blank[col].isna()]
#         mat_tmp = mat[:, df2.index.tolist()]
#         max_match = mat_tmp.argmax(1)
#         cos_match = mat_tmp.gather(1, max_match.unsqueeze(1)).data.cpu().numpy()
#         max_match = max_match.data.cpu().numpy()
#         label_lst_1, label_lst_2 = [], []
#         for i in range(len(cos_match)):
#             if cos_match[i] > 0.8:
#                 label_lst_1.append(i)
#                 label_lst_2.append(max_match[i])
#         if len(label_lst_1) == 0:
#             continue
#         label_lst_1 = df1[target_col].values[label_lst_1]
#         label_lst_2 = df2[col].values[label_lst_2]
#         pcc = abs(np.corrcoef(label_lst_1, label_lst_2)[0][1])
#         if pcc > 0.2 and len(label_lst_1) > 50:
#             col_lst.append(col)
#             class_weights.append(pcc**2)
#         del mat_tmp, max_match, cos_match
#     del mat, mat1, mat2
#     gc.collect()
#     torch.cuda.empty_cache()
#     class_weights = np.array(class_weights)
#     df_blank = load_data(args, ckpt)
#     df_blank = df_blank[col_lst]
#     
#     # datapoint selection
#     df_blank['tmp'] = df_blank[col_lst[1:]].fillna(-1).values.tolist()
#     df_blank = df_blank.loc[df_blank['tmp'].apply(lambda x: sum(x))!=-(len(col_lst)-1)].reset_index(drop=True)
#     
#     df_blank['tmp'] = df_blank[col_lst[1:]].fillna(-1).values.tolist()
#     df_blank = df_blank.loc[df_blank['tmp'].apply(
#         lambda x: 1 if (x[0]!= -1 or 
#                         np.sum(np.where(np.array(x)==-1, 0, 1)*np.where(class_weights>0.1, 10, class_weights))>1) else 0)==1].reset_index(drop=True)
#     del df_blank['tmp']
#     class_weights_new = [1.0]
#     for i, col in enumerate(col_lst[2:]):
#         if sum(~df_blank[col].isna()) < 20:
#             del df_blank[col]
#         else:
#             class_weights_new.append(class_weights[i+1])
#     class_weights = np.array(class_weights_new)
#     col_lst = df_blank.columns.tolist()
#     print(df_blank.shape)
#     
#     # 列名后处理
#     df_blank.columns = ['smiles'] + ['_'.join(item.split('_')[:-1]) for item in df_blank.columns[1:]]
#     target_col = '_'.join(target_col.split('_')[:-1])
# # =============================================================================
# #     df_blank[['smiles', target_col]].loc[~df_blank[target_col].isna()].to_csv(args.raw_path, index=False)
# # =============================================================================
#     task_type = [item.split('_')[-1] for item in col_lst[1:]]
#     
#     return df_blank, class_weights, task_type
# =============================================================================


def get_class_sizes(data: MoleculeDataset) -> List[List[float]]:
    """
    Determines the proportions of the different classes in the classification dataset.

    :param data: A classification dataset
    :return: A list of lists of class proportions. Each inner list contains the class proportions
    for a task.
    """
    targets = data.targets()

    # Filter out Nones
    valid_targets = [[] for _ in range(data.num_tasks())]
    for i in range(len(targets)):
        for task_num in range(len(targets[i])):
            if targets[i][task_num] is not None:
                valid_targets[task_num].append(targets[i][task_num])

    class_sizes = []
    for task_targets in valid_targets:
        # Make sure we're dealing with a binary classification task
        if not set(np.unique(task_targets)) <= {0, 1}:
            continue

        try:
            ones = np.count_nonzero(task_targets) / len(task_targets)
        except ZeroDivisionError:
            ones = float('nan')
            print('Warning: class has no targets')
        class_sizes.append([1 - ones, ones])

    return class_sizes


def validate_data(data_path: str) -> Set[str]:
    """
    Validates a data CSV file, returning a set of errors.

    :param data_path: Path to a data CSV file.
    :return: A set of error messages.
    """
    errors = set()

    header = get_header(data_path)

    with open(data_path) as f:
        reader = csv.reader(f)
        next(reader)  # Skip header

        smiles, targets = [], []
        for line in reader:
            smiles.append(line[0])
            targets.append(line[1:])

    # Validate header
    if len(header) == 0:
        errors.add('Empty header')
    elif len(header) < 2:
        errors.add('Header must include task names.')

    mol = Chem.MolFromSmiles(header[0])
    if mol is not None:
        errors.add('First row is a SMILES string instead of a header.')

    # Validate smiles
    for smile in tqdm(smiles, total=len(smiles)):
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            errors.add('Data includes an invalid SMILES.')

    # Validate targets
    num_tasks_set = set(len(mol_targets) for mol_targets in targets)
    if len(num_tasks_set) != 1:
        errors.add('Inconsistent number of tasks for each molecule.')

    if len(num_tasks_set) == 1:
        num_tasks = num_tasks_set.pop()
        if num_tasks != len(header) - 1:
            errors.add('Number of tasks for each molecule doesn\'t match number of tasks in header.')

    unique_targets = set(np.unique([target for mol_targets in targets for target in mol_targets]))

    if unique_targets <= {''}:
        errors.add('All targets are missing.')

    for target in unique_targets - {''}:
        try:
            float(target)
        except ValueError:
            errors.add('Found a target which is not a number.')

    return errors
