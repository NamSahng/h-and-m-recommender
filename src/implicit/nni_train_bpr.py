import argparse
import logging

import nni
from nni.utils import merge_parameter

import implicit

from scipy.sparse import coo_matrix
from implicit.evaluation import mean_average_precision_at_k

import os
import datetime

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '../../')
from src.utils.preprocess import *
logger = logging.getLogger("nni_implicit")

def get_params():
    # Training settings
    parser = argparse.ArgumentParser()

    parser.add_argument("--use_week_num", type=int, default=4)
    parser.add_argument("--use_full_item", type=bool, default=True)

    parser.add_argument("--use_gpu", type=bool, default=False)
    parser.add_argument("--model", type=str, default="als")
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--regularization", type=float, default=0.1)
    parser.add_argument("--factors", type=int, default=30)
    parser.add_argument("--verify_negative_samples", type=bool, default=True)
    
    args, _ = parser.parse_known_args()
    return args


def to_user_item_coo(df, unique_users, unique_items):
    row = df['user_id'].values
    col = df['item_id'].values    
    data = np.ones(df.shape[0])
    coo = coo_matrix((data, (row, col)), shape=(len(unique_users), len(unique_items)))
    return coo


def preprocess(trans_df, art_df, use_week_num, users2num, use_full_item):
    use_week_num += 1 # for validation 1 week
    start_date = trans_df.t_dat.max() - datetime.timedelta(7 * use_week_num)
    cur_trans_df = trans_df[trans_df['t_dat'] > start_date]

    if use_full_item == False:
        items = cur_trans_df.article_id.unique().tolist()
    else:
        items = art_df.article_id.unique().tolist()
        
    num2items = dict(list(enumerate(items)))
    items2num = dict(zip(num2items.values(), num2items.keys()))

    cur_trans_df['user_id'] = cur_trans_df.customer_id.map(users2num)
    cur_trans_df['item_id'] = cur_trans_df.article_id.map(items2num)

    train_df, val_df = split_train_valid(cur_trans_df, 0)

    coo_train = to_user_item_coo(train_df, users2num.values(), items)
    coo_val = to_user_item_coo(val_df, users2num.values(), items)
    
    csr_train = coo_train.tocsr()
    csr_val = coo_val.tocsr()
    return {
            'csr_train': csr_train,
            'csr_val': csr_val
          }


def validate(model, matrices, 
            use_gpu,
            learning_rate,
            iterations,
            regularization,
            factors,
            verify_negative_samples, 
            random_state=2022):
    csr_train, csr_val = matrices['csr_train'], matrices['csr_val']
    if model == 'als':
        cur_model = implicit.als.AlternatingLeastSquares(factors=factors, 
                                                    iterations=iterations, 
                                                    regularization=regularization, 
                                                    random_state=random_state)
    elif model == 'bpr':
        cur_model = implicit.bpr.BayesianPersonalizedRanking(use_gpu = use_gpu,
                                                        learning_rate = learning_rate,
                                                        iterations = iterations,
                                                        regularization = regularization,
                                                        factors = factors,
                                                        verify_negative_samples = verify_negative_samples,
                                                        random_state=random_state)
    elif model == 'lmf':                    
        cur_model = implicit.lmf.LogisticMatrixFactorization(factors=factors, 
                                                        regularization=regularization, 
                                                        iterations=iterations, 
                                                        random_state=random_state)
                                                        
    cur_model.fit(csr_train, show_progress=False)
    
    # The MAPK by implicit doesn't allow to calculate allowing repeated items, which is the case.
    # TODO: change MAP@12 to a library that allows repeated items in prediction
    map12 = mean_average_precision_at_k(cur_model, csr_train, csr_val, K=12, show_progress=False, num_threads=4)
    return map12


def main(args, trans_df, art_df, users2num, seed):

    use_week_num = args["use_week_num"]
    use_full_item = args["use_full_item"]

    use_gpu = args["use_gpu"]
    model = args["model"]
    learning_rate = args["learning_rate"]
    iterations = args["iterations"]
    regularization = args["regularization"]
    factors = args["factors"]
    verify_negative_samples = args["verify_negative_samples"]

    matrices = preprocess(trans_df, art_df, use_week_num, users2num, use_full_item)    
    nni.report_final_result(validate(model, matrices, 
    use_gpu = use_gpu,
    learning_rate = learning_rate,
    iterations = iterations,
    regularization = regularization,
    factors = factors,
    verify_negative_samples = verify_negative_samples,
    random_state = seed))


if __name__ == "__main__":
    seed = 2022
    folder = '../../data/'
    logger.info(os.getcwd())
    art_df = pd.read_csv(os.path.join(folder, "articles.csv"))
    cus_df = pd.read_csv(os.path.join(folder, "customers.csv"))
    trans_df = pd.read_csv(os.path.join(folder, "transactions_train.csv"))
    trans_df = make_weeknum_col(trans_df)
    trans_df['t_dat'] = pd.to_datetime(trans_df['t_dat'])
    users = cus_df.customer_id.unique().tolist()
    num2users = dict(list(enumerate(users)))
    users2num = dict(zip(num2users.values(), num2users.keys()))
    try:
        tuner_params = nni.get_next_parameter()
        params = vars(merge_parameter(get_params(), tuner_params))
        print(params)
        main(params, trans_df, art_df, users2num, seed)
    except Exception as exception:
        logger.exception(exception)
        raise
        # tcgmxr0q
        