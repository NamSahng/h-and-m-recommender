import math
import scipy.stats as stats
import numpy as np

def calc_mpg_lambda(t, t_mean, t_purch, t_pg, k, alpha, beta):
    lambda_ac = None 
    if t > (t_mean * 2):
        lambda_ac = (k + alpha) / (t_pg + beta)
    else:
        lambda_ac = (k + alpha) / (t_purch + (2*abs(t_mean-t)) + beta)
    return lambda_ac
        
def get_lambda(row):
    t_min = row['t_min']/np.timedelta64(1, 'D')
    t_max = row['t_max']/np.timedelta64(1, 'D')
    k = row['cnt']
    alpha = row['shape']
    beta = row['rate']
    t_mean = row['t_mean']
    t_purch = row['t_purch']
    t_pg_min = row['t_pg_min']/np.timedelta64(1, 'D')
    t_pg_max = row['t_pg_max']/np.timedelta64(1, 'D') 
    lambda_ac_min = calc_mpg_lambda(t_min, t_mean, t_purch, t_pg_min, k, alpha, beta)
    lambda_ac_max = calc_mpg_lambda(t_max, t_mean, t_purch, t_pg_max, k, alpha, beta)
    
    return lambda_ac_min, lambda_ac_max

def calc_hpp(rate, num_loop=10):
    rac = 0
    for i in range(1,num_loop+1):
        rac += ((rate**(i)) * math.exp(rate)) / math.factorial(i)
    return rac

def get_shape_rate(row, artinfo_train_df, col):
    cur_aid =  row[col]
    cur_dist = artinfo_train_df[artinfo_train_df[col] == cur_aid][f'{col}_dist']
    cur_dist = cur_dist[~np.isnan(cur_dist)]
    shape, _, scale = stats.gamma.fit(cur_dist) # shape, loc, scale
    rate = (1/scale)
    return shape, rate