from collections import defaultdict
import logging
import random
from typing import Dict, List, Set, Tuple, Union

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm
import numpy as np
from numba import jit
from time import time


from .data import MoleculeDataset


def generate_scaffold(mol: Union[str, Chem.Mol], include_chirality: bool = False) -> str:
    """
    Compute the Bemis-Murcko scaffold for a SMILES string.

    :param mol: A smiles string or an RDKit molecule.
    :param include_chirality: Whether to include chirality.
    :return:
    """
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)

    return scaffold


def scaffold_to_smiles(mols: Union[List[str], List[Chem.Mol]],
                       use_indices: bool = False) -> Dict[str, Union[Set[str], Set[int]]]:
    """
    Computes scaffold for each smiles string and returns a mapping from scaffolds to sets of smiles.

    :param mols: A list of smiles strings or RDKit molecules.
    :param use_indices: Whether to map to the smiles' index in all_smiles rather than mapping
    to the smiles string itself. This is necessary if there are duplicate smiles.
    :return: A dictionary mapping each unique scaffold to all smiles (or smiles indices) which have that scaffold.
    """
    scaffolds = defaultdict(set)
    for i, mol in tqdm(enumerate(mols), total=len(mols)):
        scaffold = generate_scaffold(mol)
        if use_indices:
            scaffolds[scaffold].add(i)
        else:
            scaffolds[scaffold].add(mol)

    return scaffolds


def scaffold_split(data: MoleculeDataset,
                   sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                   balanced: bool = False,
                   seed: int = 0,
                   logger: logging.Logger = None) -> Tuple[MoleculeDataset,
                                                           MoleculeDataset,
                                                           MoleculeDataset]:
    """
    Split a dataset by scaffold so that no molecules sharing a scaffold are in the same split.

    :param data: A MoleculeDataset.
    :param sizes: A length-3 tuple with the proportions of data in the
    train, validation, and test sets.
    :param balanced: Try to balance sizes of scaffolds in each set, rather than just putting smallest in test set.
    :param seed: Seed for shuffling when doing balanced splitting.
    :param logger: A logger.
    :return: A tuple containing the train, validation, and test splits of the data.
    """
    assert sum(sizes) == 1

    # Split
    train_size, val_size, test_size = sizes[0] * len(data), sizes[1] * len(data), sizes[2] * len(data)
    train, val, test = [], [], []
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

    # Map from scaffold to index in the data
    scaffold_to_indices = scaffold_to_smiles(data.mols(), use_indices=True)

    if balanced:  # Put stuff that's bigger than half the val/test size into train, rest just order randomly
        index_sets = list(scaffold_to_indices.values())
        big_index_sets = []
        small_index_sets = []
        for index_set in index_sets:
            if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
                big_index_sets.append(index_set)
            else:
                small_index_sets.append(index_set)
        random.seed(seed)
        random.shuffle(big_index_sets)
        random.shuffle(small_index_sets)
        index_sets = big_index_sets + small_index_sets
    else:  # Sort from largest to smallest scaffold sets
        index_sets = sorted(list(scaffold_to_indices.values()),
                            key=lambda index_set: len(index_set),
                            reverse=True)

    for index_set in index_sets:
        if len(train) + len(index_set) <= train_size:
            train += index_set
            train_scaffold_count += 1
        elif len(val) + len(index_set) <= val_size:
            val += index_set
            val_scaffold_count += 1
        else:
            test += index_set
            test_scaffold_count += 1

    if logger is not None:
        logger.debug(f'Total scaffolds = {len(scaffold_to_indices):,} | '
                     f'train scaffolds = {train_scaffold_count:,} | '
                     f'val scaffolds = {val_scaffold_count:,} | '
                     f'test scaffolds = {test_scaffold_count:,}')
    
    log_scaffold_stats(data, index_sets, logger=logger)

    # Map from indices to data
    train = [data[i] for i in train]
    val = [data[i] for i in val]
    test = [data[i] for i in test]

    return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)


def scaffold_similarity_split(
        data: MoleculeDataset,
        sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        balanced: bool = False,
        seed: int = 0,
        logger: logging.Logger = None) -> Tuple[
        MoleculeDataset, MoleculeDataset, MoleculeDataset]:
    assert sum(sizes) == 1

    # Split
    train_size, val_size, test_size = sizes[0] * len(data), sizes[1] * len(data), sizes[2] * len(data)
    train, val, test = [], [], []
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

    # Map from scaffold to index in the data
    scaffold_to_indices = scaffold_to_smiles(data.mols(), use_indices=True)
    mat_similarity = get_similarity_matrix(data.mols())

    if balanced:  # Put stuff that's bigger than half the val/test size into train, rest just order randomly
        index_sets = list(scaffold_to_indices.values())
        index_sets = merge_similar_scaffold(index_sets, mat_similarity)
        big_index_sets = []
        small_index_sets = []
        for index_set in index_sets:
            if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
                big_index_sets.append(index_set)
            else:
                small_index_sets.append(index_set)
        random.seed(seed)
        random.shuffle(big_index_sets)
        random.shuffle(small_index_sets)
        index_sets = big_index_sets + small_index_sets
    else:  # Sort from largest to smallest scaffold sets
        assert 0
        index_sets = sorted(list(scaffold_to_indices.values()),
                            key=lambda index_set: len(index_set),
                            reverse=True)

    for index_set in index_sets:
        if len(train) + len(index_set) <= train_size:
            train += index_set
            train_scaffold_count += 1
        elif len(val) + len(index_set) <= val_size:
            val += index_set
            val_scaffold_count += 1
        else:
            test += index_set
            test_scaffold_count += 1

    if logger is not None:
        logger.debug(f'Total scaffolds = {len(scaffold_to_indices):,} | '
                     f'train scaffolds = {train_scaffold_count:,} | '
                     f'val scaffolds = {val_scaffold_count:,} | '
                     f'test scaffolds = {test_scaffold_count:,}')
    
    log_scaffold_stats(data, index_sets, logger=logger)

    # Map from indices to data
    train = [data[i] for i in train]
    val = [data[i] for i in val]
    test = [data[i] for i in test]

    return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)


def get_similarity_matrix(mols):
    print('get similarity matrix...')
    from rdkit import DataStructs
    from rdkit.Chem import AllChem
    from tqdm import trange
    
    def similarity(fp1, fp2):
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    mat = np.ones((len(mols), len(mols)))

    arr = [AllChem.GetMorganFingerprintAsBitVect(
            mol, 2, nBits=2048, useChirality=False) for mol in mols]
    for i in trange(len(arr)-1):
        for j in range(i+1, len(arr)):
            score = similarity(arr[i], arr[j])
            mat[i][j] = score
            mat[j][i] = score
    print('done.')
    return mat


def merge_similar_scaffold(index_lst, mat):
    print('total scaffold num =', len(index_lst))
    
    ## 拆分类内
# =============================================================================
#     new_index_lst = []
#     c = 0
#     for i in range(len(index_lst)):
#         if len(index_lst[i]) / mat.shape[0] > 0.03:
#             flg, s = calc_similarity_in_scaffolds(index_lst[i], mat)
#             if flg:
#                 lst1, lst2 = split_index_lst(index_lst[i])
#                 new_index_lst.append(set(lst1))
#                 new_index_lst.append(set(lst2))
#                 c += 1
#                 continue
#         new_index_lst.append(index_lst[i])
#     index_lst = new_index_lst
#     print('split num =', c)
# =============================================================================
    
    ## 合并类间
    merge_index_lst = np.zeros(len(index_lst))
    idx = 1
    for i in range(len(index_lst)-1):
        for j in range(i+1, len(index_lst)):
            flg, s = calc_similarity_between_scaffolds(
                    index_lst[i], index_lst[j], mat)
            if flg:
                if merge_index_lst[i] == 0 and merge_index_lst[j] == 0:
                    merge_index_lst[i] = idx
                    merge_index_lst[j] = idx
                    idx += 1
                elif merge_index_lst[i] != 0 and merge_index_lst[j] == 0:
                    merge_index_lst[j] = merge_index_lst[i]
                elif merge_index_lst[i] == 0 and merge_index_lst[j] != 0:
                    merge_index_lst[i] = merge_index_lst[j]
                else:
                    if merge_index_lst[i] != merge_index_lst[j]:
                        idx1 = merge_index_lst[i]
                        idx2 = merge_index_lst[j]
                        merge_index_lst = [item if item != idx1 else idx2 for item in merge_index_lst]
                    else:
                        continue
    new_index_lst = []
    for i in range(len(index_lst)):
        idx = merge_index_lst[i]
        if idx == 0:
            new_index_lst.append(index_lst[i])
        elif idx != -1:
            lst = []
            for j in range(len(index_lst)):
                if merge_index_lst[j] == idx:
                    lst += index_lst[j]
                    merge_index_lst[j] = -1
            new_index_lst.append(lst)
    print('total new scaffold num =', len(new_index_lst))
    return new_index_lst
                        
def calc_similarity_between_scaffolds(idx_lst1, idx_lst2, mat, T=0.3):
    s = []
    if len(idx_lst1) < len(idx_lst2):
        for i in idx_lst1:
            s.append(mat[i, list(idx_lst2)])
    else:
        for i in idx_lst2:
            s.append(mat[i, list(idx_lst1)])
    s = np.mean(s)
# =============================================================================
#     x = np.array(list(idx_lst1)).repeat(len(idx_lst2))
#     y = np.array(list(idx_lst2))
#     y = np.concatenate([y]*len(idx_lst1))
#             
#     s = mat[x, y].mean()
# =============================================================================
    if s > T:
        return True, s
    return False, s


def calc_similarity_in_scaffolds(idx_lst1, mat):
    s = []
    for i in idx_lst1:
        s.append(mat[i, list(idx_lst1)])
    s = np.mean(s)
    if s < 0.3:
        return True, s
    return False, s


def split_index_lst(lst):
    import random
    random.seed(2020)
    lst = list(lst)
    random.shuffle(lst)
    return lst[:int(len(lst)//2)], lst[int(len(lst)//2):]


def log_scaffold_stats(data: MoleculeDataset,
                       index_sets: List[Set[int]],
                       num_scaffolds: int = 10,
                       num_labels: int = 20,
                       logger: logging.Logger = None) -> List[Tuple[List[float], List[int]]]:
    """
    Logs and returns statistics about counts and average target values in molecular scaffolds.

    :param data: A MoleculeDataset.
    :param index_sets: A list of sets of indices representing splits of the data.
    :param num_scaffolds: The number of scaffolds about which to display statistics.
    :param num_labels: The number of labels about which to display statistics.
    :param logger: A Logger.
    :return: A list of tuples where each tuple contains a list of average target values
    across the first num_labels labels and a list of the number of non-zero values for
    the first num_scaffolds scaffolds, sorted in decreasing order of scaffold frequency.
    """
    # print some statistics about scaffolds
    target_avgs = []
    counts = []
    for index_set in index_sets:
        data_set = [data[i] for i in index_set]
        targets = [d.targets for d in data_set]
        targets = np.array(targets, dtype=np.float)
        target_avgs.append(np.nanmean(targets, axis=0))
        counts.append(np.count_nonzero(~np.isnan(targets), axis=0))
    stats = [(target_avgs[i][:num_labels], counts[i][:num_labels]) for i in range(min(num_scaffolds, len(target_avgs)))]

    if logger is not None:
        logger.debug('Label averages per scaffold, in decreasing order of scaffold frequency,'
                     f'capped at {num_scaffolds} scaffolds and {num_labels} labels: {stats}')

    return stats
