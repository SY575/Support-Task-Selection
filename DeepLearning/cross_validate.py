import warnings
warnings.filterwarnings('ignore')
from rdkit import RDLogger  
RDLogger.DisableLog('rdApp.*')  
from argparse import Namespace
from logging import Logger
import os
from typing import Tuple

import numpy as np
import pandas as pd
from chemprop.train.run_training import run_training
from chemprop.data.utils import get_task_names
from chemprop.utils import makedirs
from chemprop.parsing import parse_train_args, modify_train_args
from chemprop.utils import create_logger

def cross_validate(args: Namespace, logger: Logger = None) -> Tuple[float, float]:
    """k-fold cross validation"""
    info = logger.info if logger is not None else print
    # Initialize relevant variables
    init_seed = args.seed
    save_dir = args.save_dir

    # Run training on different random seeds for each fold
    all_scores = []
    # all_individual_test_scores = []
    for fold_num in range(args.num_folds):
        info(f'Fold {fold_num}')
        args.seed = init_seed + fold_num
        args.save_dir = os.path.join(save_dir, f'fold_{fold_num}')
        makedirs(args.save_dir)
        model_scores = run_training(args, logger)
        all_scores.append(model_scores)
    all_scores = np.array(all_scores)
    # Report results
    info(f'{args.num_folds}-fold cross validation')
    task_names = get_task_names(args.data_path)

    # Report scores for each fold
    for fold_num, scores in enumerate(all_scores):
        info(f'Seed {init_seed + fold_num} ==> test {args.metric} = {np.nanmean(scores):.6f}')

        if args.show_individual_scores:
            info(f'Seed {init_seed + fold_num} ==> test {task_names[0]} {args.metric} = {scores:.6f}')

    # Report scores across models
    avg_scores = np.nanmean(all_scores)  # average score for each model across tasks
    mean_score, std_score = np.nanmean(avg_scores), np.nanstd(avg_scores)
    # info(f'Overall test {args.metric} = {mean_score:.6f} +/- {std_score:.6f}')
    info(f'Overall test {task_names[0]} {args.metric} = '
                 f'{np.nanmean(all_scores):.6f} +/- {np.nanstd(all_scores):.6f}')

    return mean_score, std_score



if __name__ == '__main__':
    n_repeat = 5
    score_lst = []
    for i_repeat in range(n_repeat):
        args = parse_train_args()
        args.data_path = args.data_path.replace('temp', f'temp_{args.target_col}')
        args.raw_path = args.raw_path.replace('STL', f'STL_{args.target_col}')
        
        fn = args.target_col.split('_')[0]
        col = '_'.join(args.target_col.split('_')[1:])
        df = pd.read_csv(f'../data_final/{fn}.csv')
        df = df[[df.columns[0], col]]
        df.loc[~df[col].isna()].to_csv(args.raw_path, index=False)
        
        args.seed += i_repeat * 100 + args.seed
        args.save_dir = f'./ckpt_{args.target_col}'

        modify_train_args(args)
        logger = create_logger(name=f'train_{i_repeat}', save_dir=args.save_dir, quiet=args.quiet)
        class_type = args.target_col[0]
        args.datasets = {key:value for key, value in args.datasets.items() if key[0] == class_type}
        mean_auc_score, std_auc_score = cross_validate(args, logger)
        score_lst.append(mean_auc_score)
    print(score_lst)
    print(n_repeat, f'repeats score: {np.nanmean(score_lst):.5f} +/- {np.nanstd(score_lst):.5f}')

