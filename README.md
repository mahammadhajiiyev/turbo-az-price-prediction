<div align="center">

# 🚗 Turbo.az Car Price Prediction

**Predicting used car prices in Azerbaijan with machine learning**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-02569B)](https://lightgbm.readthedocs.io/)
[![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mahammadhajiiyev/turbo-az-price-prediction/blob/main/Turbo_az_project.ipynb)

</div>

---

## 📌 Overview

[Turbo.az](https://turbo.az) is Azerbaijan's largest online car marketplace. This project builds a regression model that estimates the **market price of a used car** from its listing details — brand, model, year, mileage, engine, transmission, condition and more.

The final **LightGBM** model explains **~92.5% of the price variance (R² = 0.925)** on unseen test data.

## 📊 Dataset

| | |
|---|---|
| **Source** | [Kaggle — Turbo.az cars project](https://www.kaggle.com/datasets/sehriyarmemmedli/turboaz-cars-project) |
| **Raw size** | 653,721 listings × 56 columns |
| **After cleaning** | 244,263 unique listings |
| **Used for training** | Random sample of 100,000 listings (80/20 train-test split) |
| **Target** | `price_y` — car price |

### Features used

| Group | Features |
|---|---|
| **Car identity** | Brand (`Marka`), model (`Model`), body type (`Ban növü`), production year, number of seats |
| **Technical** | Engine displacement, fuel type (`Yanacaq novu`), transmission (`Sürətlər qutusu`), drivetrain (`Ötürücü`) |
| **Condition & history** | Mileage (`Yürüş`), number of owners (`Sahiblər`), accident history (`Qəzalı`), damage / paint state (`Vəziyyəti`), new or used (`Yeni`) |
| **Listing** | City, color (`Rəng`), target market (`Hansı bazar üçün yığılıb`), barter, dealer (`salon`), VIP / featured flags |

## 🔧 Workflow

```mermaid
flowchart LR
    A[Raw data<br/>653K rows] --> B[Cleaning<br/>duplicates &<br/>irrelevant columns]
    B --> C[Preprocessing<br/>Pipeline +<br/>ColumnTransformer]
    C --> D[Linear Regression<br/>baseline]
    C --> E[Decision Tree<br/>+ RandomizedSearchCV]
    C --> F[LightGBM<br/>best model]
```

1. **Data cleaning** — removed scraping metadata, URLs and image links, unified mileage units, dropped redundant columns and duplicate listings.
2. **Preprocessing** — a single scikit-learn `Pipeline`:
   - *Numeric:* `StandardScaler` → `SimpleImputer(median)` → `PolynomialFeatures(degree=2)`
   - *Categorical:* `OneHotEncoder(handle_unknown='ignore')` → `SimpleImputer(most_frequent)`
3. **Modeling** — three models trained and compared with R², MAE and 3-fold cross-validation.

## 🏆 Results

| Model | Train R² | Test R² | Notes |
|---|:---:|:---:|---|
| Linear Regression (baseline) | 0.851 | 0.834 | CV R² = 0.824 · MAE ≈ 6,457 |
| Decision Tree Regressor | — | — | Tuned with `RandomizedSearchCV` |
| **LightGBM Regressor** ✅ | **0.960** | **0.925** | 500 trees · learning rate 0.05 |

LightGBM improves the test R² by **~9 points** over the linear baseline while keeping the gap between train and test small — the model generalizes well.

### Example prediction

```python
# 2012 Hyundai Accent · sedan · 1.4 L petrol · automatic
# 165,000 km · official dealer car · no accidents · not repainted
full_pipeline_lgbm.predict(sample_input)
# → ≈ 16,040
```

## 🚀 How to run

1. Click the **Open in Colab** badge above.
2. Upload your Kaggle API key (`kaggle.json`) when prompted — the dataset downloads automatically.
3. Run all cells.

To run locally:

```bash
pip install pandas numpy scikit-learn lightgbm matplotlib seaborn kaggle
jupyter notebook Turbo_az_project.ipynb
```

## 📁 Project structure

```
turbo-az-price-prediction/
├── Turbo_az_project.ipynb   # data cleaning, EDA, modeling, evaluation
└── README.md
```

## 🔮 Future improvements

- Hyperparameter tuning for LightGBM (`num_leaves`, `min_child_samples`, `learning_rate`)
- Train on the full 244K cleaned dataset instead of a 100K sample
- Log-transform the target to reduce the influence of very expensive cars
- Deploy as a small web app (e.g. Streamlit) where users enter car details and get a price estimate

## 👤 Author

**Mahammad Hajiyev** · [@mahammadhajiiyev](https://github.com/mahammadhajiiyev)
