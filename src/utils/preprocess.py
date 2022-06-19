import pandas as pd
import numpy as np

def make_weeknum_col(transactions:pd.DataFrame):
    """apply week number on dataframe"""
    min_date = transactions.t_dat.min()
    max_date = transactions.t_dat.max()
    dates = [i.date() for i in pd.date_range(min_date, max_date)]
    dates = pd.DataFrame(dates , columns=['t_dat'])
    dates['day_num'] = np.array(dates.index)[::-1]
    dates['week_num'] = dates['day_num'] // 7
    dates = dates[['t_dat','week_num']]
    dates['t_dat'] = dates['t_dat'].apply(str)
    transactions = pd.merge(transactions, dates, left_on='t_dat', right_on='t_dat', how='left')
    return transactions

def split_train_valid(transactions:pd.DataFrame, split_week_num:int=0):
    """
    split train valid data with week number
    week0 is latest week
    """
    assert 'week_num' in transactions.columns
    valid_tr = transactions[transactions.week_num == split_week_num]
    train_tr = transactions[transactions.week_num > split_week_num]
    return train_tr, valid_tr

def valid2submission(valid_tr:pd.DataFrame):
    """validation dataframe to submission format"""
    valid_tr = pd.DataFrame(valid_tr.groupby('customer_id')['article_id'].apply(list)).reset_index()
    valid_tr['article_id'] = valid_tr['article_id'].map(lambda x: '0'+' 0'.join(str(x)[1:-1].split(', ')))
    return valid_tr
