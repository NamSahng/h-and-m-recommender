# H & M Reccommender

- from: https://github.com/NamSahng/h-and-m-recommender


- Data: 
  - download from: https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data

## Structure
```
h-and-m-recommender
├── data
│   ├── articles.csv
│   ├── cusstomers.csv
│   ├── sample_submission.csv
│   ├── transactions_train.csv
│   └── images
│       └── 010
│       └── ...
└── src
    ├── buyitagain
    └── implicit
    └── trending_product_weekly
    └── ...
```
## Experiments
- buyitagain
  - Implementation of Modified Poisson-Gamma Model (MPG) on H&M dataset.
- implicit
  - ALS, BPR, LMF Experiments on H&M dataset.
- trending_product_weekly
  - Follow exisiting kaggle repo and modifiy it using RCP & MPG.

&nbsp;


## References
- Buy it Again
  - paper: Rahul Bhagat, et al. 2018. Buy It Again: Modeling Repeat Purchase Recommendations
  - https://assets.amazon.science/40/e5/89556a6341eaa3d7dacc074ff24d/buy-it-again-modeling-repeat-purchase-recommendations.pdf


- trending product weekly
  - https://www.kaggle.com/code/byfone/h-m-trending-products-weekly
  - https://www.kaggle.com/code/hervind/h-m-faster-trending-products-weekly/notebook
  - https://www.kaggle.com/code/byfone/h-m-purchases-in-a-row/notebook


- Metrics
  - https://www.kaggle.com/code/kaerunantoka/h-m-how-to-calculate-map-12
