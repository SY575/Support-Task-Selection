import warnings
warnings.filterwarnings('ignore')
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
import pickle
from sklearn import metrics
import scipy
import os
from random import shuffle
from rdkit.Chem import MACCSkeys
import lightgbm as lgb

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

datasets = {
            '1tox21': 'classification',
# =============================================================================
#             '1toxcast': 'classification',
#             '1sider': 'classification',
#             '1clintox': 'classification',
#             '1hERG': 'classification',
# =============================================================================
            }
d = '1tox21'
p = '../SMILES2VES.pkl'
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

def task_selection(target_col, train_smiles):
    df_blank = pd.DataFrame({'smiles':[]})
    for dataset in datasets.keys():
        cols = pd.read_csv('../data/'+dataset+'.csv', nrows=1).rename(
        columns={'SMILES':'smiles'}).columns.tolist()
        df0 = pd.read_csv('../data/'+dataset+'.csv', 
                          dtype={col:np.float16 for col in cols[1:]}).rename(
                              columns={'SMILES':'smiles'})
        df0.columns = ['smiles'] + [f'{dataset}_{col.split("(")[0]}' for col in df0.columns[1:].tolist()]
        df_blank = pd.merge(df_blank, df0, on='smiles', how='outer')
    del df0
    assert target_col in df_blank, f'Invalid target_col: {target_col}'
    
    train_smiles = list(train_smiles)
    if len(train_smiles) > 2000:
        shuffle(train_smiles)
        demo_smiles = train_smiles[:2000]
    else:
        demo_smiles = train_smiles
    X = np.array([smiles2vec(s) for s in demo_smiles if smiles2vec(s) is not None])
    col_lst = ['smiles', target_col]
    class_weights = [1.]
    
    result = {}
    for dataset in datasets:
        cols = pd.read_csv(f'../data/{dataset}.csv', nrows=1).rename(
        columns={'SMILES':'smiles'}).columns.tolist()
        cols = cols[1:]
        for label in cols:
            label = label.split("(")[0]
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
    keys = np.array([item for item in keys if item != target_col])
    pcc = np.array([abs(scipy.stats.spearmanr(
        result[target_col], result[keys[i]]).correlation) for i in range(len(keys))])
    T = 0.35
    col_lst += list(keys[np.where(pcc>T)])
    class_weights += list(pcc[np.where(pcc>T)])
    print(list(keys[np.where(pcc>T)]), list(pcc[np.where(pcc>T)]))
    class_weights = np.array(class_weights)
    df_blank = df_blank[col_lst]
    
    task_type = [datasets[dataset] for item in col_lst[1:]]
    
    return df_blank, class_weights, task_type


task = datasets[d]
tmp_lst = {'lgb':[], 'rf':[]}
for target_col in pd.read_csv(
        f'../data/{d}.csv', nrows=1).columns.tolist()[1:]:
    data_raw = pd.read_csv(f'../data/{d}.csv').rename(
        columns={'SMILES':'smiles'})[['smiles', target_col]]
    data_raw = data_raw.loc[~data_raw[target_col].isna()].reset_index(drop=True)
    target_col = f'{d}_{target_col}'
    target_col = target_col.split('(')[0]

    data_raw = data_raw['smiles'].values.tolist()
    sizes = [0.7, 0.1, 0.2]
    train_size = int(sizes[0] * len(data_raw))
    train_val_size = int((sizes[0] + sizes[1]) * len(data_raw))
    
    train_smiles = set(data_raw[:train_size])
    val_smiles = set(data_raw[train_size:train_val_size])
    test_smiles = set(data_raw[train_val_size:])
    
    df, class_weights, task_type = task_selection(
        target_col, train_smiles)
    cols = df.columns.tolist()[2:]
    
# =============================================================================
#     df = pd.read_csv(f'../data/{d}.csv')
#     df.columns = ['smiles'] + [f'{d}_{item}' for item in df.columns.tolist()[1:]]
#     cols = []
# =============================================================================
# =============================================================================
#     
# =============================================================================
    new_df = df[['smiles', target_col]]
    new_df['feature'] = new_df['smiles'].apply(
        lambda x: smiles2vec(x))
    new_df = new_df.loc[~new_df['feature'].isna()].reset_index(drop=True)
    X = np.array(new_df['feature'].values.tolist())
    for item in cols:
        dataset = item.split('_')[0]
        label = '_'.join(item.split('_')[1:])
        fp = f'../ckpt_STL/clf_{dataset}_{label}.pkl'
        try:
            clf = pickle.load(open(fp, 'rb'))
            new_df[f'{dataset}_{label}_pred'] = clf.predict_proba(X)[:, 1]
        except:
            try:
                fp = f'../ckpt_STL/clf_{dataset}_{label.split("(")[0]}.pkl'
                clf = pickle.load(open(fp, 'rb'))
                new_df[f'{dataset}_{label}_pred'] = clf.predict(X)
            except:
                print(fp)
    pred_columns = new_df.columns.tolist()[3:]
    
    result_lst = {'lgb':[], 'rf':[]}
    for n_repeat in range(3):
        data_raw = pd.read_csv(
            f'../data/{d}.csv').rename(
        columns={'SMILES':'smiles'}).sample(frac=1.0)['smiles'].values.tolist()
        sizes = [0.6, 0.2, 0.2]
        train_size = int(sizes[0] * len(data_raw))
        train_val_size = int((sizes[0] + sizes[1]) * len(data_raw))
        
        train_smiles = set(data_raw[:train_size])
        val_smiles = set(data_raw[train_size:train_val_size])
        test_smiles = set(data_raw[train_val_size:])
        
        
        train_df = new_df.loc[new_df.smiles.apply(
            lambda x: 1 if x in train_smiles else 0)==1].reset_index(drop=True)
        val_df = new_df.loc[new_df.smiles.apply(
            lambda x: 1 if x in val_smiles else 0)==1].reset_index(drop=True)
        val_df = val_df.loc[~val_df[target_col].isna()].reset_index(drop=True)
        
        test_df = new_df.loc[new_df.smiles.apply(
            lambda x: 1 if x in test_smiles else 0)==1].reset_index(drop=True)
        test_df = test_df.loc[~test_df[target_col].isna()].reset_index(drop=True)
        
        
        train_data = np.array(train_df['feature'].values.tolist())
        train_data = np.concatenate((train_data, train_df[pred_columns].values), -1)
        train_label = train_df[target_col].fillna(2).values
        
        val_data = np.array(val_df['feature'].values.tolist())
        val_data = np.concatenate((val_data, val_df[pred_columns].values), -1)
        val_label = val_df[target_col].values# .fillna(2).values
        
        test_data = np.array(test_df['feature'].values.tolist())
        test_data = np.concatenate((test_data, test_df[pred_columns].values), -1)
        test_label = test_df[target_col].values# .fillna(2).values
        
        if task == 'classification':
            clf = lgb.LGBMClassifier(
                n_estimators=1000,
                )
            clf.fit(train_data, train_label, 
                    eval_set=[(val_data, val_label)],
                    verbose=False, early_stopping_rounds=50,
                    )
            test_pred = clf.predict_proba(test_data)[:, 1]
            result = metrics.roc_auc_score(test_label, test_pred)
        else:
            clf = lgb.LGBMRegressor(n_estimators=1000)
            clf.fit(train_data, train_label, 
                    eval_set=[(val_data, val_label)],
                    verbose=False, early_stopping_rounds=50)
            test_pred = clf.predict(test_data)
            result = metrics.mean_squared_error(test_label, test_pred)**0.5
        result_lst['lgb'].append(result)
        
        if task == 'classification':
            clf = RandomForestClassifier()
            clf.fit(train_data, train_label)
            test_pred = clf.predict_proba(test_data)[:, 1]
            result = metrics.roc_auc_score(test_label, test_pred)
        else:
            clf = RandomForestRegressor()
            clf.fit(train_data, train_label)
            test_pred = clf.predict(test_data)
            result = metrics.mean_squared_error(test_label, test_pred)**0.5
        result_lst['rf'].append(result)
    result = np.mean(result_lst['lgb'])
    if task == 'classification':
        print(f'{target_col} AUC: {result:.3f}')
    else:
        print(f'{target_col} RMSE: {result:.3f}')
    tmp_lst['lgb'].append(np.mean(result_lst['lgb']))
    tmp_lst['rf'].append(np.mean(result_lst['rf']))
print(f'lgb: {np.mean(tmp_lst["lgb"]):.5f}')
print(f'rf: {np.mean(tmp_lst["rf"]):.5f}')
pickle.dump(SMILES2VES, open(p, 'wb'))
