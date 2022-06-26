import os
import gc
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import sys
sys.path.insert(0, './')
from src.buyitagain.preprocess_bia import *
from src.buyitagain.utils_bia import *


def func(x, a, b, c, d):
    return a / np.sqrt(x) + b * np.exp(-c*x) + d

if __name__ == "__main__":
    print('started')
    folder = './data/'
    output_dir = './data/submission'
    os.makedirs(output_dir, exist_ok=True)
    art_df = pd.read_csv(os.path.join(folder, "articles.csv"))
    cus_df = pd.read_csv(os.path.join(folder, "customers.csv"))
    trans_df = pd.read_csv(os.path.join(folder, "transactions_train.csv"))
    train_df = trans_df 
    train_df['t_dat'] = pd.to_datetime(train_df.t_dat)
    gb = train_df.groupby(['customer_id', 'article_id'])['t_dat'].nunique()
    inx = gb[gb>1].index
    rp_train_df = train_df.set_index(['customer_id', 'article_id'])
    rp_train_df = rp_train_df.loc[inx].copy().sort_index()
    rp_train_df['shift_dat'] = rp_train_df.groupby(level=[0,1])['t_dat'].shift(1)
    rp_train_df['dist'] = (rp_train_df['t_dat'] - rp_train_df['shift_dat']).dt.days
    dist = rp_train_df.loc[rp_train_df['dist'].notna(), 'dist'].values
    vc = rp_train_df['dist'].value_counts()
    vc = vc[1:]
    popt, pcov = curve_fit(func, vc.index, vc.values)
    N = 12
    last_ts = train_df['t_dat'].max()
    tmp = train_df[['t_dat']].copy()
    tmp['dow'] = tmp['t_dat'].dt.dayofweek
    tmp['ldbw'] = tmp['t_dat'] - pd.TimedeltaIndex(tmp['dow'] - 1, unit='D')
    tmp.loc[tmp['dow'] >=2 , 'ldbw'] = tmp.loc[tmp['dow'] >=2 , 'ldbw'] + \
                            pd.TimedeltaIndex(np.ones(len(tmp.loc[tmp['dow'] >=2])) * 7, unit='D')
    train_df['ldbw'] = tmp['ldbw'].values
    weekly_sales = train_df.drop('customer_id', axis=1)\
                        .groupby(['ldbw', 'article_id'])\
                        .count().reset_index()
    weekly_sales = weekly_sales.rename(columns={'t_dat': 'count'})
    weekly_sales = weekly_sales[['ldbw', 'article_id', 'count']]
    train_df = train_df.merge(weekly_sales, on=['ldbw', 'article_id'], how = 'left')
    weekly_sales = weekly_sales.reset_index().set_index('article_id')
    last_week_sales = weekly_sales.loc[weekly_sales['ldbw']==last_ts, ['count']]
    train_df = train_df.merge(last_week_sales, on='article_id', suffixes=("", "_targ"))
    train_df['count_targ'].fillna(0, inplace=True)
    train_df['quotient'] = train_df['count_targ'] / train_df['count']
    purchase_dict = {}
    tmp = train_df.copy()
    tmp['x'] = ((last_ts - tmp['t_dat']) / np.timedelta64(1, 'D')).astype(int)
    tmp['dummy_1'] = 1 
    tmp['x'] = tmp[["x", "dummy_1"]].max(axis=1)
    a, b, c, d = popt
    tmp['y'] = a / np.sqrt(tmp['x']) + b * np.exp(-c*tmp['x']) - d
    tmp['dummy_0'] = 0 
    tmp['y'] = tmp[["y", "dummy_0"]].max(axis=1)
    tmp['value'] = tmp['quotient'] * tmp['y'] 
    
    print('tpw done')
    print('#'*60)
    
    del train_df, trans_df; gc.collect()

    trans_df = pd.read_csv(os.path.join(folder, "transactions_train.csv"))
    train_df = trans_df

    col = 'article_id'
    rcp_threshold = 0.0
    min_num_purchased = 0
    use_cols = ['t_dat', 'customer_id', 'article_id']
    train_df = train_df[use_cols]
    art_df['idxgrp_idx_prdtyp'] = art_df['index_group_name'] + '_' + art_df['index_name'] + '_' + art_df['product_type_name']
    use_cols = ['article_id', 'prod_name', 'idxgrp_idx_prdtyp']
    sample_art_df = art_df[use_cols]
    train_df = pd.merge(train_df, sample_art_df, how='left', on='article_id')
    train_df['t_dat'] = pd.to_datetime(train_df['t_dat'])
    train_df['trans_idx'] = train_df.index
    col_train_df = train_df.drop_duplicates(subset=["t_dat", "customer_id", col], keep='last')
    col_train_df = make_time_interval_col(col_train_df, col)
    col_g_df = groupby_cid_artinfo(col_train_df, col)
    col_g_df = make_rcp_df(col_g_df, col)
    rp_aid = get_repeat_purchasable(col_g_df, rcp_threshold, denom_customer_num=1)
    print(len(rp_aid))
    new_tmp = pd.merge(tmp, rp_aid, on='article_id', how='left')

    new_tmp['rcp_mean'] = new_tmp.rcp.fillna(new_tmp.rcp.mean())
    new_tmp['new_value'] = new_tmp['value'] * new_tmp['rcp_mean']

    new_value_col = f'new_value'
    cut_value = new_tmp[new_value_col].quantile(0.0) 
    cut_new_tmp = new_tmp.loc[new_tmp[new_value_col] >= cut_value]
    
    cut_new_tmp['rank'] = cut_new_tmp.groupby("customer_id")[new_value_col]\
                                        .rank("dense", ascending=False)
    cut_new_tmp = cut_new_tmp.loc[cut_new_tmp['rank'] <= 12]
    
    purchase_df = cut_new_tmp.sort_values(['customer_id', new_value_col], ascending = False).reset_index(drop = True)
    
    purchase_df['prediction'] = '0' + purchase_df['article_id'].astype(str) + ' '
    purchase_df = purchase_df.groupby('customer_id').agg({'prediction': sum}).reset_index()
    purchase_df['prediction'] = purchase_df['prediction'].str.strip()
    
    pred_df = pd.merge(purchase_df, cus_df['customer_id'], on='customer_id', how='right')
    pred_df.fillna('', inplace=True)

    pred_df.to_csv(os.path.join(output_dir, 'trending_product_weekly_rcp_01.csv'), index=False)
