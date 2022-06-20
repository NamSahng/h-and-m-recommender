import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

if __name__ == "__main__":

    folder = './data/'
    output_dir = './data/submission'
    os.makedirs(output_dir, exist_ok=True)
    art_df = pd.read_csv(os.path.join(folder, "articles.csv"))
    cus_df = pd.read_csv(os.path.join(folder, "customers.csv"))
    trans_df = pd.read_csv(os.path.join(folder, "transactions_train.csv"))

    train_df = trans_df 

    train_df['t_dat'] = pd.to_datetime(train_df.t_dat)


    # gb: (고객, 제품)별 날짜에 unique한 거래 횟수
    # inx: 2번 이상 거래된 (고객, 제품)인덱스
    gb = train_df.groupby(['customer_id', 'article_id'])['t_dat'].nunique()
    inx = gb[gb>1].index

    # rp_train_df: 2번 이상 거래된 
    rp_train_df = train_df.set_index(['customer_id', 'article_id'])
    rp_train_df = rp_train_df.loc[inx].copy().sort_index()

    rp_train_df['shift_dat'] = rp_train_df.groupby(level=[0,1])['t_dat'].shift(1)
    rp_train_df['dist'] = (rp_train_df['t_dat'] - rp_train_df['shift_dat']).dt.days
    dist = rp_train_df.loc[rp_train_df['dist'].notna(), 'dist'].values

    vc = rp_train_df['dist'].value_counts()
    # 하루 이상의 제품-고객 거래의 시간 간격 개수 
    vc = vc[1:]

    def func(x, a, b, c, d):
        return a / np.sqrt(x) + b * np.exp(-c*x) + d
    # vc.index : 재구매 일수
    # vc.values : 재구매 일수 별 횟수
    popt, pcov = curve_fit(func, vc.index, vc.values)

    N = 12

    last_ts = train_df['t_dat'].max()

    tmp = train_df[['t_dat']].copy()
    # day of week
    tmp['dow'] = tmp['t_dat'].dt.dayofweek
    #  Monday=0, ... ,Sunday=6

    # last day of billing week
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

    # 마지막주 상품의 팔린 횟수를 count_targ으로 설정
    last_week_sales = weekly_sales.loc[weekly_sales['ldbw']==last_ts, ['count']]
    train_df = train_df.merge(last_week_sales, on='article_id', suffixes=("", "_targ"))
    train_df['count_targ'].fillna(0, inplace=True)

    train_df['quotient'] = train_df['count_targ'] / train_df['count']


    purchase_dict = {}

    # x : 마지막 날짜와 시간 간격, 마지막 날짜의 거래의 시간간격은 1로 설정
    tmp = train_df.copy()
    tmp['x'] = ((last_ts - tmp['t_dat']) / np.timedelta64(1, 'D')).astype(int)
    tmp['dummy_1'] = 1 
    tmp['x'] = tmp[["x", "dummy_1"]].max(axis=1)

    # 위에서 fitting한 식에 대하여, 재구매가 일어날 가능도 처럼 사용?
    # y = 재구매가 일어날 수 있는 상대적 가능도

    a, b, c, d = popt
    tmp['y'] = a / np.sqrt(tmp['x']) + b * np.exp(-c*tmp['x']) - d

    # quotient = 마지막 주에 해당 물품 팔린 횟수 / 해당 주에 해당 물품 팔린 횟수 
    # value = 해당 상품 예상 재구매 가능도 * 마지막 주에 해당상품 팔린 횟수 / 해당 주에 해당 물품 팔린 횟수 

    tmp['dummy_0'] = 0 
    tmp['y'] = tmp[["y", "dummy_0"]].max(axis=1)
    tmp['value'] = tmp['quotient'] * tmp['y'] 

    # 그리고 value를 전체 거래에 대하여 합침
    tmp = tmp.groupby(['customer_id', 'article_id']).agg({'value': 'sum'})
    tmp = tmp.reset_index()


    # 100 이상의 값으로 자르고, 랭킹 12개 이하로

    tmp = tmp.loc[tmp['value'] > 100]
    tmp['rank'] = tmp.groupby("customer_id")["value"].rank("dense", ascending=False)
    tmp = tmp.loc[tmp['rank'] <= 12]

    purchase_df = tmp.sort_values(['customer_id', 'value'], ascending = False).reset_index(drop = True)

    purchase_df['prediction'] = '0' + purchase_df['article_id'].astype(str) + ' '
    purchase_df = purchase_df.groupby('customer_id').agg({'prediction': sum}).reset_index()
    purchase_df['prediction'] = purchase_df['prediction'].str.strip()

    pred_df = pd.merge(purchase_df, cus_df['customer_id'], on='customer_id', how='right')
    pred_df.fillna('', inplace=True)

    pred_df.to_csv(os.path.join(output_dir, 'trending_product_weekly_followed.csv'), index=False)