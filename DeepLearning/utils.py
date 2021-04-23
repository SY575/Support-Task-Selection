import pandas as pd
import numpy as np
import os
import scipy
from random import shuffle
from rdkit import Chem
from rdkit.Chem import MACCSkeys
import pickle

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

def task_selection(target_col, datasets, train_smiles):
    # merge datasets
    df_blank = pd.DataFrame({'smiles':[]})
    for dataset in datasets.keys():
        cols = pd.read_csv('../data_final/'+dataset+'.csv', nrows=1).columns.tolist()
        df0 = pd.read_csv('../data_final/'+dataset+'.csv', 
                          dtype={col:np.float16 for col in cols[1:]})
        df0.columns = ['smiles'] + [f'{dataset}_{col}' for col in df0.columns[1:].tolist()]
        df_blank = pd.merge(df_blank, df0, on='smiles', how='outer')
    del df0
    assert target_col in df_blank, f'Invalid target_col: {target_col}'
    
    # sample
    train_smiles = list(train_smiles)
    if len(train_smiles) > 2000:
        shuffle(train_smiles)
        demo_smiles = train_smiles[:2000]
    else:
        demo_smiles = train_smiles
    X = np.array([smiles2vec(s) for s in demo_smiles])
    
    # support task selection
    # Step 1. Prediction
    col_lst = ['smiles', target_col]
    class_weights = [1.]
    result = {}
    for dataset in datasets:
        cols = pd.read_csv(f'../data_final/{dataset}.csv', nrows=1).columns.tolist()
        cols = cols[1:]
        for label in cols:
            fp = f'./ckpt_STL/clf_{dataset}_{label}.pkl'
            try:
                clf = pickle.load(open(fp, 'rb'))
                result[f'{dataset}_{label}'] = clf.predict_proba(X)[:, 1]
            except:
                try:
                    clf = pickle.load(open(fp, 'rb'))
                    result[f'{dataset}_{label}'] = clf.predict(X)
                except:
                    print(fp)
    # Step 2. Calculating spearman R
    keys = sorted(list(result.keys()))
    keys = np.array([item for item in keys if item != target_col])
    score = np.array([abs(scipy.stats.spearmanr(
        result[target_col], result[keys[i]]).correlation) for i in range(len(keys))])
    T = 0.35
    col_lst += list(keys[np.where(score>T)])
    class_weights += list(score[np.where(score>T)])
    class_weights = np.array(class_weights)
    df_blank = df_blank[col_lst]
    
    # datapoint selection
    df_blank['tmp'] = df_blank[col_lst[1:]].fillna(-1).values.tolist()
    df_blank = df_blank.loc[df_blank['tmp'].apply(
        lambda x: sum(x))!=-(len(col_lst)-1)].reset_index(drop=True)
    
    df_blank['tmp'] = df_blank[col_lst[1:]].fillna(-1).values.tolist()
    df_blank = df_blank.loc[df_blank['tmp'].apply(
        lambda x: 1 if (x[0]!= -1 or 
                        np.sum(np.where(np.array(x)==-1, 0, 1)*\
                               np.where(class_weights>0.1, 10, 
                                        class_weights))>1) else 0)==1]
    df_blank = df_blank.reset_index(drop=True)
    del df_blank['tmp']
    class_weights_new = [1.0]
    for i, col in enumerate(col_lst[2:]):
        if sum(~df_blank[col].isna()) < 100:
            del df_blank[col]
        else:
            class_weights_new.append(class_weights[i+1])
    class_weights = np.array(class_weights_new)
    col_lst = df_blank.columns.tolist()
    # print(df_blank.shape)
    
    task_type = [datasets[dataset] for item in col_lst[1:]]
    
    return df_blank, class_weights, task_type