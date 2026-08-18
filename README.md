# Credit Risk Prediction for Micro-Loans in Morocco

A machine learning project that predicts whether a micro-loan applicant is **low-risk** or **high-risk**, using survey data on financial habits and digital activity instead of traditional credit histories.

Many people in Morocco have no formal credit record, which makes them hard to assess with standard scoring methods. This project explores whether alternative signals — savings behavior, income sources, digital payments, account ownership — can fill that gap and help lenders serve underbanked borrowers more fairly.


---

## Dataset

- **Source:** [Global Findex Database 2021](https://www.worldbank.org/en/publication/globalfindex) (World Bank) — Morocco subset
- **Size:** 1,000 respondents, 114 original columns
- **Used for modeling:** 22 features after cleaning and encoding

Features cover three areas:

| Category | Examples |
|---|---|
| Demographics | age, gender, education, employment status |
| Financial behavior | income level, bank account, savings, income sources |
| Digital access | internet access, digital payments, merchant payments |

We also built a few combined features, such as a `digital_activity` score and a `financial_stability` score.

**Target:** borrowers who recently borrowed but had no savings and no stable income sources were labeled high-risk.
The classes are imbalanced — about 89% low-risk vs. 11% high-risk.

---

## Approach

1. **Clean the data** — drop irrelevant survey columns, fill missing ages, fix data types.
2. **Engineer features** — build composite scores for digital activity, income diversity, and financial stability.
3. **Prevent leakage** — remove the variables that were used to define the target label.
4. **Split** — 80% training / 20% testing, stratified to keep the class balance realistic.
5. **Rebalance** — apply SMOTETomek to the training set only, so the test set stays untouched.
6. **Train and compare** three models.

---

## Models

| Model | Why it's here |
|---|---|
| **Logistic Regression** | Main model — simple, fast, and easy to explain to lenders and regulators |
| **Random Forest** | Ensemble comparator that captures non-linear patterns |
| **XGBoost** | Strong performer on tabular data, good at catching rare cases |

---

## Results

Because high-risk cases are rare, I looked beyond plain accuracy and relied on ROC-AUC, PR-AUC, F1-score, and balanced accuracy.

| Model | ROC-AUC |
|---|---|
| Logistic Regression | **0.94** |
| XGBoost | 0.93 |
| Random Forest | 0.91 |

**Takeaways:**

- Logistic Regression came out best overall and stays the most interpretable.
- XGBoost was strongest on PR-AUC, meaning it's slightly better at catching actual high-risk borrowers.
- Random Forest was competitive but produced more false alarms.
- The most influential features were **financial stability**, **account ownership**, **remittances**, and **digital payment activity**.
- Most errors came from borderline cases — people with mixed signals, like decent digital activity but shaky income.

---

## System Architecture

**Training (offline):** data cleaning and feature engineering in a Jupyter notebook → train and evaluate models → save the best one.

**Deployment (online):**

```
User → Web interface (Vercel) → Backend API (Render) → Saved model → Risk score + decision
```

The backend applies the exact same preprocessing used during training, so predictions stay consistent. All predictions are logged for monitoring.

---

## Ethics

- The dataset is fully anonymized — no personally identifiable information.
- Gender and income disparities were monitored in the predictions.
- Balanced accuracy was tracked to make sure both classes are treated fairly.
- Feature importance analysis was used to check for proxy discrimination.

---



