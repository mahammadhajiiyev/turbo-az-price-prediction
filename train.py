"""Train the Turbo.az price model and export everything the Streamlit app needs.

Usage:
    python train.py                 # downloads the dataset with the Kaggle API if needed
    python train.py --csv cars.csv  # use an already downloaded CSV

Outputs (in ./model):
    model.joblib      - fitted scikit-learn pipeline (preprocessing + LightGBM)
    meta.json         - dropdown options, brand -> model mapping, metrics, error band
    price_stats.csv   - median price per brand / model / year for the app's charts
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"
DATASET = "sehriyarmemmedli/turboaz-cars-project"
USD_EUR_TO_AZN = 1.7  # same conversion rate as the notebook

NUM_COLS = ["production_year", "engine_displacement_num", "horsepower", "Yürüş", "Sahiblər", "Yerlərin sayı"]
BIN_COLS = ["city", "barter", "salon", "Qəzalı", "Yeni"]
CAT_COLS = ["Marka", "Model", "Ban növü", "Hansı bazar üçün yığılıb", "Rəng",
            "Sürətlər qutusu", "Vəziyyəti", "Ötürücü", "Yanacaq novu"]
FEATURES = NUM_COLS + BIN_COLS + CAT_COLS
TARGET = "price"


def download(csv_path: Path) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    print(f"Downloading {DATASET} from Kaggle...")
    subprocess.run([sys.executable, "-m", "kaggle", "datasets", "download", "-d", DATASET,
                    "-p", str(DATA_DIR), "--unzip"], check=True)
    if not csv_path.exists():
        sys.exit(f"Expected {csv_path} after download, but it is missing.")


def to_int(series: pd.Series, drop: str) -> pd.Series:
    cleaned = series.astype("string").str.replace(drop, "", regex=True).str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Same cleaning steps as Turbo_az_project.ipynb, plus outlier filtering."""
    out = pd.DataFrame()
    out["city"] = (df["city"] == "bakı").astype(int)
    out["production_year"] = pd.to_numeric(df["production_year"], errors="coerce")
    out["engine_displacement_num"] = pd.to_numeric(df["engine_displacement_num"], errors="coerce")
    # "2.0 L / 184 a.g. / Dizel" -> 184 (horsepower is not used in the notebook, but adds signal)
    out["horsepower"] = pd.to_numeric(
        df["Mühərrik"].astype("string").str.extract(r"(\d+)\s*a\.g\.")[0], errors="coerce")
    out["Yürüş"] = to_int(df["Yürüş"], r"km|\s")
    out["Sahiblər"] = to_int(df["Sahiblər"], r" və daha çox")
    out["Yerlərin sayı"] = to_int(df["Yerlərin sayı"], r"\+")
    out["barter"] = (df["barter"] == "Barter mümkündür").astype(int)
    out["salon"] = (df["salon"] == "Salon").astype(int)
    out["Qəzalı"] = (df["Qəzalı"] == "Bəli").astype(int)
    out["Yeni"] = (df["Yeni"] == "Bəli").astype(int)
    for col in CAT_COLS:
        if col == "Yanacaq novu":
            out[col] = df["Mühərrik"].astype("string").str.split("/").str[-1].str.strip()
        else:
            out[col] = df[col].astype("string").str.strip()

    price = pd.to_numeric(df["price_y"], errors="coerce")
    out[TARGET] = np.where(df["currency_y"].isin(["USD", "EUR"]), price * USD_EUR_TO_AZN, price)

    # scikit-learn imputers expect np.nan, not pandas' nullable pd.NA
    out[NUM_COLS] = out[NUM_COLS].astype(float)
    out[CAT_COLS] = out[CAT_COLS].astype(object).where(out[CAT_COLS].notna(), np.nan)

    out = out.drop_duplicates()
    out = out[out["production_year"].between(1960, 2026)
              & out[TARGET].between(1_000, 1_000_000)
              & (out["Yürüş"].isna() | (out["Yürüş"] < 1_500_000))
              & out["Marka"].notna() & out["Model"].notna()]
    return out.reset_index(drop=True)


def build_pipeline() -> TransformedTargetRegressor:
    preprocessing = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUM_COLS + BIN_COLS),
        ("cat", make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
        ), CAT_COLS),
    ])
    lgbm = LGBMRegressor(
        n_estimators=1500, learning_rate=0.05, num_leaves=127, min_child_samples=20,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=42, verbosity=-1,
    )
    model = make_pipeline(preprocessing, lgbm)
    # Predicting log(price) keeps very expensive cars from dominating the error
    return TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DATA_DIR / "cars.csv")
    args = parser.parse_args()

    if not args.csv.exists():
        download(args.csv)

    print("Loading and cleaning data...")
    df = clean(pd.read_csv(args.csv, low_memory=False))
    print(f"  {len(df):,} listings after cleaning")

    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training LightGBM...")
    model = build_pipeline().fit(X_train, y_train)
    pred = model.predict(X_test)
    ratio = pred / y_test.to_numpy()
    metrics = {
        "r2": round(float(r2_score(y_test, pred)), 4),
        "mae": round(float(mean_absolute_error(y_test, pred)), 0),
        "median_abs_pct_error": round(float(np.median(np.abs(ratio - 1)) * 100), 2),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }
    print(f"  Test R2 = {metrics['r2']}, MAE = {metrics['mae']:,.0f} AZN, "
          f"median error = {metrics['median_abs_pct_error']}%")

    print("Refitting on all data and saving...")
    model = build_pipeline().fit(X, y)
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "model.joblib", compress=3)

    brand_models = (df.groupby("Marka")["Model"]
                    .apply(lambda s: s.value_counts().index.tolist()).to_dict())
    brand_counts = df["Marka"].value_counts()

    # Typical specs per brand/model so the app can pre-fill the form when a model is picked
    def typical(group: pd.DataFrame) -> dict:
        spec = {c: group[c].mode().iat[0] for c in
                ("Ban növü", "Sürətlər qutusu", "Ötürücü", "Yanacaq novu") if group[c].notna().any()}
        for c in ("engine_displacement_num", "horsepower", "Yerlərin sayı"):
            if group[c].notna().any():
                spec[c] = float(group[c].median())
        spec["year"] = int(group["production_year"].median())
        return spec
    model_specs = {f"{b}|{m}": typical(g) for (b, m), g in df.groupby(["Marka", "Model"])}
    meta = {
        "metrics": metrics,
        # 80% of test predictions fall within this band around the true price
        "error_band": [round(float(np.quantile(1 / ratio, q)), 3) for q in (0.1, 0.9)],
        "brands": brand_counts.index.tolist(),
        "brand_models": brand_models,
        "model_specs": model_specs,
        "options": {c: df[c].value_counts().index.tolist() for c in CAT_COLS if c not in ("Marka", "Model")},
        "defaults": {
            "engine_displacement_num": float(df["engine_displacement_num"].median()),
            "horsepower": int(df["horsepower"].median()),
            "Yerlərin sayı": int(df["Yerlərin sayı"].median()),
        },
        "year_range": [int(df["production_year"].min()), int(df["production_year"].max())],
        "rows": int(len(df)),
    }
    (MODEL_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    stats = (df.groupby(["Marka", "Model", "production_year"])[TARGET]
             .agg(median_price="median", listings="size").reset_index())
    stats = stats[stats["listings"] >= 3]
    stats.to_csv(MODEL_DIR / "price_stats.csv", index=False)

    size_mb = (MODEL_DIR / "model.joblib").stat().st_size / 1e6
    print(f"Done. model.joblib = {size_mb:.1f} MB, {len(stats):,} price-stat rows")


if __name__ == "__main__":
    main()
