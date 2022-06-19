import pandas as pd
import numpy as np


def make_time_interval_col(train_df, col='article_id'):
    assert col in ('article_id', 'prod_name', 'idxgrp_idx_prdtyp')
    train_df = train_df.set_index(['customer_id', col])
    train_df = train_df.sort_values('t_dat')
    train_df = train_df.sort_index()

    train_df[f'{col}_lastdate'] = train_df.groupby(level=[0,1])['t_dat'].shift(1)
    train_df[f'{col}_dist'] = (train_df['t_dat'] - train_df[f'{col}_lastdate']).dt.days

    train_df = train_df.reset_index()
    return train_df

def groupby_cid_artinfo(transaction_df, col):
    grouped = transaction_df.groupby([f'{col}', 'customer_id'])['t_dat'].count().reset_index()
    grouped = grouped.rename(columns={'t_dat': 'cnt'})
    grouped = grouped.sort_values('cnt', ascending= False)
    return grouped

def make_rcp_df(grouped_df, col):
    denom = pd.DataFrame(grouped_df.groupby(f'{col}')['customer_id'].nunique()).reset_index()
    denom = denom.rename(columns={'customer_id': 'denom_customer'})

    numer = pd.DataFrame(grouped_df[grouped_df.cnt > 1].groupby(f'{col}')['customer_id'].nunique()).reset_index()
    numer = numer.rename(columns={'customer_id': 'num_customer'})

    rcp_df = pd.merge(denom, numer, on= f'{col}')
    rcp_df['rcp'] = rcp_df['num_customer']/rcp_df['denom_customer']
    return rcp_df

def get_repeat_purchasable(rcp_df, rcp_threshold = 0.2, denom_customer_num=None, denom_customer_prop=None):
    # denom_customer_num: number of people who bouhgt the product at least once
    if denom_customer_prop and denom_customer_num is None:
        denom_customer_num = rcp_df.denom_customer.quantile(denom_customer_prop)
    elif denom_customer_prop is None and denom_customer_num is None:
        denom_customer_num = 0
    print(f'재구매가능 상품의 최소 고객수: {denom_customer_num}')
    repeat_purchasable_df = rcp_df[(rcp_df.rcp > rcp_threshold) & 
                                (rcp_df.denom_customer >= denom_customer_num)]
    return repeat_purchasable_df

def get_rp_cus_artinfo(train_df, col, min_repeat_time=4):
    rp_cus_artinfo = train_df.groupby(['customer_id', col]).agg(
        repeated = pd.NamedAgg(column='t_dat', aggfunc='count'),
        trans_idxes = pd.NamedAgg(column='trans_idx', aggfunc=list)
        ).reset_index()
    rp_cus_artinfo = rp_cus_artinfo[rp_cus_artinfo.repeated >=  min_repeat_time]
    return rp_cus_artinfo

def get_rp_cid_col_df(train_df, col):
    rp_cid_col_df = train_df.groupby(['customer_id', col]).agg(
        t_mean=pd.NamedAgg(column=f'{col}_dist', aggfunc=np.nanmean),
        t_purch=pd.NamedAgg(column=f'{col}_dist', aggfunc=np.nansum),
        t_last=pd.NamedAgg(column='t_dat', aggfunc='max'),
        t_first=pd.NamedAgg(column='t_dat', aggfunc='min'),
        cnt=pd.NamedAgg(column=f'{col}_dist', aggfunc='count'))
    rp_cid_col_df = rp_cid_col_df.reset_index()
    return rp_cid_col_df

def make_date_info_col(val_df, rp_cid_col_df):
    val_df['t_dat'] = pd.to_datetime(val_df.t_dat)
    rp_cid_col_df['cur_date_min'] = val_df.t_dat.min()
    rp_cid_col_df['cur_date_max'] = val_df.t_dat.max()
    rp_cid_col_df['t_min'] = rp_cid_col_df['cur_date_min'] - rp_cid_col_df['t_last']
    rp_cid_col_df['t_max'] = rp_cid_col_df['cur_date_max'] - rp_cid_col_df['t_last']
    rp_cid_col_df['t_pg_min'] = rp_cid_col_df['cur_date_min'] - rp_cid_col_df['t_first']
    rp_cid_col_df['t_pg_max'] = rp_cid_col_df['cur_date_max'] - rp_cid_col_df['t_first']
    return rp_cid_col_df
