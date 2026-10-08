# Who did an SMS campaign really move? Uplift modelling on X5 RetailHero

**Question.** A grocery retailer sent a promotional SMS to a random half of its loyalty customers. Which customers bought *because of* the message, and which would have bought anyway?

**Data.** X5 RetailHero uplift dataset (X5 Retail Group, public competition): 200,039 customers with a randomised SMS treatment, a purchase outcome and about 4 million transactions before the campaign. The data is not included here; download it via `scikit-uplift` (`fetch_x5`) or the competition page, and put `uplift_train.csv`, `clients.csv`, `products.csv` and `purchases.csv` into `data/`.

## Approach
- Weekly purchase history becomes each customer's baseline ("what they normally do").
- The SMS effect is estimated by segment: treated vs control customers, each compared with their own history, with Monte Carlo / bootstrap intervals.
- Segments are found on one half of customers and confirmed on the other half.
- Benchmarked against tuned S-, T-, X- and DR-learners, class transformation and a causal forest (EconML).

## Results
**Overall effect:** +3.75 percentage points more customers bought because of the SMS (95% interval 3.0–4.5). About 19 in 20 buyers in the SMS group would have bought anyway.

**Segments (confirmed on the held-out half):**

| Segment | Share | Extra buyers per 100 messages |
|---|---|---|
| Small baskets, regular loyalty-point users | 4% | +11.1 (7.7 to 15.0) |
| Small baskets, rarely use points | 16% | +4.0 to +4.3 |
| Larger baskets, usual frequency | 74% | +3.5 (2.7 to 4.3) |
| Very frequent big-basket shoppers | 6% | +1.3 (−0.8 to 3.4) |

**Benchmark (uplift@30%, held-out 30%, 10 splits):**

| Features | My method | Best benchmark | Verdict |
|---|---|---|---|
| 6 basic features | 0.0587 | 0.0572 (tuned S-learner) | on par (difference not significant) |
| 117 engineered features | 0.0723 (forest variant) | 0.0764 (tuned X-learner) | on par (difference not significant) |

## Limitations
- With 117 features, the simple segment-averaging variant fell behind tuned ML models; the forest variant stayed on par.
- Profit figures in the reports use illustrative costs (€0.08 per SMS, €6 margin per extra buyer).

## How this was built
Method design, research questions and validation choices are my own. I used an AI assistant (Claude) to help write and refactor the code and documentation.

## How to run
```
pip install -r requirements.txt
python x5_basic_features.py     # 6 features, method vs tuned learners
python x5_rich_features.py      # 117 features, segments tree, forest, report table
```
Both scripts expect the data in the folder set by `DATA` at the top of the file. Code comments are partly in Russian.
