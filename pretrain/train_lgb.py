import pandas as pd
import numpy as np
import lightgbm as lgb
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
import pickle
import os
from sklearn import metrics
os.makedirs('../ckpt_STL', exist_ok=True)
from rdkit.Chem import MACCSkeys

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
# datasets = os.listdir('../data/')
# =============================================================================
# datasets = ['1toxcast']
# task = 'classification'
# =============================================================================
# =============================================================================
# datasets = ['4pcba']
# task = 'classification'
# =============================================================================
datasets = ['5qm8', '5qm7']
task = 'regression'
# =============================================================================
# datasets = ['1clintox', '1hERG', '1sider', '1tox21', '1toxcast']
# task = 'classification'
# =============================================================================
# =============================================================================
# datasets = ['2BBB', '2esol', '2logD', '2logP', '2solubility']
# task = 'regression'
# =============================================================================
# =============================================================================
# datasets = ['3CYP1A2I', '3CYP2C9I', '3CYP2C19I', '3CYP2D6I', '3CYP3A4I']
# task = 'classification'
# =============================================================================
assert task in ['regression', 'classification']
for dataset in datasets:
    print(dataset)
    result_lst = []
    cols = pd.read_csv(f'../data/{dataset}.csv', nrows=1).columns.tolist()
    SMILES = cols[0]
    cols = cols[1:]
    for label in cols:
        df = pd.read_csv(f'../data/{dataset}.csv', 
                         usecols=[SMILES, label])
        df = df.loc[~df[label].isna()].reset_index(drop=True)
        print(label, df.shape[0])
        df = df.loc[df[SMILES].apply(
            lambda x: 1 if smiles2vec(x) is not None else 0)==1]
        X = np.array([smiles2vec(s) for s in df[SMILES]])
        y = df[label].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=554)
        if task == 'regression':
            clf = lgb.LGBMRegressor(n_estimators=5000)
            clf.fit(X_train, y_train, eval_set=[(X_test, y_test)],
                    verbose=100, early_stopping_rounds=50, 
                    eval_metric='mse')
            pred = clf.predict(X_test)
            result = metrics.mean_squared_error(y_test, pred)
            result_lst.append(result)
            print(dataset, f'MSE {result:.3f}')
        else:
            clf = lgb.LGBMClassifier(n_estimators=5000)
            clf.fit(X_train, y_train, eval_set=[(X_test, y_test)],
                    verbose=100, early_stopping_rounds=50, 
                    eval_metric='auc')
            pred = clf.predict_proba(X_test)[:, 1]
            result = metrics.roc_auc_score(y_test, pred)
            result_lst.append(result)
            print(dataset, f'AUC {result:.3f}')
        new_label = label.split('(')[0]
        pickle.dump(clf, open(f'../ckpt_STL/clf_{dataset}_{new_label}.pkl', 'wb'))
    print(dataset, round(np.mean(result_lst), 3))

