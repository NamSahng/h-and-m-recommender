# Trending Product Weekly


## Prediction
$$ V_{A_{i}C_{j}} = \Sigma_{m=1}^{n} value_{m} $$
- $m$: full transaction history for products traded within the last week
- $ V_{A_{i}C_{j}} <= 100 $ are not used.
- rank values and predict top 12
  

### value
- $ value_{m} = q_{m} \times f(t_{m}) $


### $f(t)$ 
- curve fiton formula below with time interval data $t_{A_{i},C_{j}}$
$$ f(t) = {{a}\over{\sqrt{t_{A_{i},C_{j}}}}} + b\exp(-c \times t_{A_{i},C_{j}}) + d  $$
- $ t_{A_{i},C_{j}}$: time interval of Customer Ci on product Ai 

### quotients
- if Ai has been sold on last week
$$ q_{m} = {{\text{num of sales on Ai in week k}}\over{\text{num of sales on Ai in last week of train set }}} $$
- else:
$$ q_{m} = 0 $$


## Modification Trials

- $ value_{m} = q_{m} \times f(t_{m}) \times MPG $
- $ value_{m} = q_{m} \times MPG $
- $ value_{m} = q_{m} \times f(t_{m}) \times RCP $


## Result 
- original
    - Validation(last week) with value cut off at 100:
        - MAP@12: 0.022690
    - test private
        - MAP@12 : 0.02165
    - test public 
        - MAP@12 : 0.02137
    - Validation(last week) without value cut off at 100:
        - MAP@12: 0.021569
- original + RCP 
    - Validation(last week)
        - MAP@12: 0.023080
    - test private
        - MAP@12 : 0.02125
    - test public 
        - MAP@12 : 0.02114


## References:
- https://www.kaggle.com/code/byfone/h-m-trending-products-weekly
- https://www.kaggle.com/code/hervind/h-m-faster-trending-products-weekly/notebook
- https://www.kaggle.com/code/byfone/h-m-purchases-in-a-row/notebook
- https://github.com/NamSahng/h-and-m-recommender/tree/main/src/buyitagain

