# Buy it again


Implementation of Buy it again on H&M kaggle dataset

- detailed version: https://github.com/NamSahng/h-and-m-recommender/blob/main/src/buyitagain/notebooks/buyitagain_customer_article_detailed.ipynb
- simplified version: https://github.com/NamSahng/h-and-m-recommender/blob/main/src/buyitagain/notebooks/buyitagain_simplified.ipynb

<br>

## Problem Formulation
- Customer $C_j$ purchased a product $A_i$ $k$ times in the past with time intervals $t_1, t_2, t_3, \cdots, t_k$
- Purchase Probalility Density:
  $$ P_{A_i}(t_{k+1}=t|t_{1},t_{2}, ... t\_{k}) $$
- Decomposition of Purchase Probalility Density
  $$ P_{A_i}(t_{k+1}|t_{1},t_{2}, ... t_{k}) \approx Q(A_{i}) \times R_{A_i}(t_{k+1}|t_{1},t_{2}, ... t_{k}, A_{i} = 1) $$
  - $Q(A_{i})$: repeat purchase probability of a customer buying a product a $(k+1)^{th}$ time given that they have bought it k times
  - $R_{A_i}$: the distribution of $t_{k+1}$, conditioned on the customer repurchasing that product; indicated by $A_{i} = 1$


## Repeated Customer Probability Model
- Formula:\
$$ P_{A_i}(t_{k+1}|t_{1},t_{2}, ... t_{k}) \approx Q(A_{i}) \approx RCP_{A_{i}} = {{\text{number of customers who bought prodcut Ai more than once}}\over{\text{number of customers who bought prodcut Ai at least once}}} $$

- recommend by: $RCP_{A_i} > r_{threshold}$

### cf) Aggregate Time Distribution Model
- Formula:
  $$ P_{A_i}(t_{k+1}|t_{1},t_{2}, ... t_{k}) \approx Q(A_{i}) \times R_{A_i}(t_{k+1}|t_{1},t_{2}, ... t_{k}, A_{i} = 1) $$
where,
    - $ R_{A_i}(t) $ is log-normal dist. with random variable t (time interval)
     $$ R_{A_i}(t) = \ln \mathcal{N}(t ; \bar{\mu_{i}}; \bar{\sigma_{i}}) = {{1}\over{t \bar{\sigma_{i}}\sqrt{2\pi} }} \exp \left[- {{(\ln{t} - \bar{\mu_{i}} ) ^{2}}\over{2\bar{\sigma_{i}}}^{2} }\right], \ \ t >0 $$
        - $\bar{\mu_{i}}, \bar{\sigma_{i}}$: parameters determined by the maximum-likelihood principle 
        - $ Q(A_{i}) $: fixed constant for all products $A_i$ at any given time t. 
        - $A_{i}$: products that satisfies $RCP_{A_i} > r_{threshold}$ are deemed.

## Modified Poisson-Gamma Model (MPG)

- Formula:
  $$ P_{A_i}(t_{k+1}|t_{1},t_{2}, ... t_{k}) \approx Q(A_{i}) \times R_{A_i}(t_{k+1}|t_{1},t_{2}, ... t_{k}, A_{i} = 1) $$

- $R_{A_{i}, C_{j}}(t)$: is a homogeneous Poisson’s process with repeat purchase rate $\lambda$. (they assume that successive repeat purchases are not correlated with each other.)
    $$R_{A_{i}, C_{j}}(t)=\sum_{m=1}^{\infty} \frac{\lambda_{A_{i}, C_{j}}^{m} \exp \left(\lambda_{A_{i}, C_{j}}\right)}{m !}, \ \ t>0$$ 

- m: number of expected future purchases
- gamma prior on $\ \lambda $, assume that $\ \lambda $ across all customers follows a Gamma distribution with shape $\ \alpha $ and rate $\ \beta $.
    - when $t > 2 \times t_{mean}$
$$\lambda_{A_{i}, C_{j}}=\frac{k+\alpha_{A_{i}}} {t+\beta_{A_{i}}}, \ \ t>0$$ 
        - In the PG model, the parameters of the product-specific gamma distributions are estimated in an empirical fashion by fitting them to the maximum likelihood estimates of the purchase rates of repeat purchasing customers.
        - $\alpha_{A_{i}} $, $\ \beta_{A_{i}} $ : the shape and rate parameters of the gamma prior of product ${A_{i}}$
        - k : the number of purchases of product $\ {A_{i}} $ by customer $\ {C_{j}} $
        - t : elapsed time between the **first purchase** of product $\ {A_{i}} $ by customer $\ {C_{j}} $ and the current time

    - when $t < 2 \times t_{mean}$: 
      $$ \lambda_{A_{i},C_{j}} = {{k+\alpha_{A_{i}}} \over {t_{purch} + 2  |t_{mean} \  - \ t| + \beta_{A_{i}}}} , \ \, t>0 $$
        - $t_{mean}$: estimated mean repeat purchase time interval between successive purchases of product Ai by customer Cj 
        - $t_{purch}$ :elapsed time interval between the first and last purchase of product Ai by customer Cj 
        - t : the elapsed time interval between the **last purchase** of product Ai by customer Cj and the current time
- $ Q(A\_{i}) $: same with ATD/RCP


## References:
- paper: Rahul Bhagat, et al. 2018. Buy It Again: Modeling Repeat Purchase Recommendations
  - https://assets.amazon.science/40/e5/89556a6341eaa3d7dacc074ff24d/buy-it-again-modeling-repeat-purchase-recommendations.pdf
