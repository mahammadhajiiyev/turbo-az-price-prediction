"""Streamlit app: estimate a used car's market price in Azerbaijan.

Run locally:  streamlit run app.py
The model and its metadata are produced by train.py.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

MODEL_DIR = Path(__file__).parent / "model"
ACCENT = "#D62839"

st.set_page_config(page_title="Avtomobil Qiymət Proqnozu", page_icon="🚗", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_DIR / "model.joblib")


@st.cache_data
def load_meta() -> dict:
    return json.loads((MODEL_DIR / "meta.json").read_text(encoding="utf-8"))


@st.cache_data
def load_stats() -> pd.DataFrame:
    return pd.read_csv(MODEL_DIR / "price_stats.csv")


def fmt_azn(value: float) -> str:
    return f"{value:,.0f}".replace(",", " ") + " AZN"


st.markdown(f"""
<style>
.block-container {{ padding-top: 2rem; max-width: 1150px; }}
.hero {{ background: linear-gradient(135deg, {ACCENT} 0%, #8E1B26 100%); color: #fff;
        padding: 28px 32px; border-radius: 18px; margin-bottom: 24px; }}
.hero h1 {{ color: #fff; margin: 0 0 6px 0; font-size: 2.1rem; }}
.hero p {{ margin: 0; opacity: .9; font-size: 1.05rem; }}
.price-card {{ background: linear-gradient(135deg, {ACCENT} 0%, #8E1B26 100%); color: #fff;
              padding: 26px; border-radius: 18px; text-align: center;
              box-shadow: 0 10px 30px rgba(214, 40, 57, .25); }}
.price-card .label {{ font-size: .95rem; opacity: .85; letter-spacing: .04em; text-transform: uppercase; }}
.price-card .price {{ font-size: clamp(1.7rem, 3vw, 2.6rem); font-weight: 800; margin: 6px 0;
                      white-space: nowrap; }}
.price-card .range {{ font-size: 1rem; opacity: .9; }}
.section-title {{ font-weight: 700; font-size: 1.05rem; margin: 6px 0 2px 0; }}
.footnote {{ color: #6B7280; font-size: .85rem; }}
@media (max-width: 640px) {{
  .hero {{ padding: 20px; }}
  .hero h1 {{ font-size: 1.5rem; }}
  .hero p {{ font-size: .95rem; }}
}}
</style>
""", unsafe_allow_html=True)

meta = load_meta()
model = load_model()
stats = load_stats()
opts = meta["options"]
metrics = meta["metrics"]

st.markdown("""
<div class="hero">
  <h1>🚗 Avtomobil Qiymət Proqnozu</h1>
  <p>Avtomobilin məlumatlarını daxil edin. Maşın öyrənməsi modeli Azərbaycan bazarındakı
  200 000-dən çox elan əsasında təxmini bazar qiymətini hesablayacaq.</p>
</div>
""", unsafe_allow_html=True)


def apply_model_defaults() -> None:
    """Pre-fill technical fields with the typical values of the chosen model."""
    spec = meta["model_specs"].get(f"{st.session_state.brand}|{st.session_state.model}", {})
    for key, field in (("body", "Ban növü"), ("gearbox", "Sürətlər qutusu"),
                       ("drive", "Ötürücü"), ("fuel", "Yanacaq novu")):
        if field in spec:
            st.session_state[key] = spec[field]
    if "engine_displacement_num" in spec:
        st.session_state.engine = round(spec["engine_displacement_num"], 1)
    if "horsepower" in spec:
        st.session_state.hp = int(spec["horsepower"])
    if "Yerlərin sayı" in spec:
        st.session_state.seats = int(spec["Yerlərin sayı"])
    if "year" in spec:
        st.session_state.year = spec["year"]


def on_brand_change() -> None:
    st.session_state.model = meta["brand_models"][st.session_state.brand][0]
    apply_model_defaults()


if "brand" not in st.session_state:
    st.session_state.brand = "Hyundai"
    st.session_state.model = "Accent"
    apply_model_defaults()
    st.session_state.mileage = 150_000

tab_predict, tab_about = st.tabs(["🔮 Qiymət hesabla", "📊 Model haqqında"])

with tab_predict:
    st.markdown('<div class="section-title">Avtomobil</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    brand = c1.selectbox("Marka", meta["brands"], key="brand", on_change=on_brand_change)
    car_model = c2.selectbox("Model", meta["brand_models"][brand], key="model",
                             on_change=apply_model_defaults)
    year_min, year_max = meta["year_range"]
    year = c3.number_input("Buraxılış ili", year_min, year_max, step=1, key="year")

    c1, c2, c3 = st.columns(3)
    body = c1.selectbox("Ban növü", opts["Ban növü"], key="body")
    color = c2.selectbox("Rəng", opts["Rəng"])
    market = c3.selectbox("Hansı bazar üçün yığılıb", opts["Hansı bazar üçün yığılıb"])

    st.markdown('<div class="section-title">Texniki göstəricilər</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    engine = c1.number_input("Mühərrik həcmi (L)", 0.0, 10.0, step=0.1, format="%.1f", key="engine")
    hp = c2.number_input("At gücü (a.g.)", 0, 1500, step=5, key="hp")
    fuel = c3.selectbox("Yanacaq növü", opts["Yanacaq novu"], key="fuel")

    c1, c2, c3 = st.columns(3)
    gearbox = c1.selectbox("Sürətlər qutusu", opts["Sürətlər qutusu"], key="gearbox")
    drive = c2.selectbox("Ötürücü", opts["Ötürücü"], key="drive")
    seats = c3.number_input("Yerlərin sayı", 1, 9, step=1, key="seats")

    st.markdown('<div class="section-title">Vəziyyət</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    mileage = c1.number_input("Yürüş (km)", 0, 1_500_000, step=5_000, key="mileage")
    condition = c2.selectbox("Vəziyyəti", opts["Vəziyyəti"])
    owners = c3.selectbox("Sahiblərin sayı", ["Məlum deyil", 0, 1, 2, 3, "4 və daha çox"], index=2)

    c1, c2, c3, c4, c5 = st.columns(5)
    in_baku = c1.checkbox("Bakıda satılır", value=True)
    is_new = c2.checkbox("Yenidir", value=(mileage == 0))
    accident = c3.checkbox("Qəzalıdır")
    dealer = c4.checkbox("Salondan satılır")
    barter = c5.checkbox("Barter mümkündür")

    st.write("")
    clicked = st.button("🔍 Qiyməti hesabla", type="primary", width="stretch")

    if clicked:
        owners_num = {"Məlum deyil": None, "4 və daha çox": 4}.get(owners, owners)
        row = pd.DataFrame([{
            "production_year": year, "engine_displacement_num": engine, "horsepower": hp or None,
            "Yürüş": mileage, "Sahiblər": owners_num, "Yerlərin sayı": seats,
            "city": int(in_baku), "barter": int(barter), "salon": int(dealer),
            "Qəzalı": int(accident), "Yeni": int(is_new),
            "Marka": brand, "Model": car_model, "Ban növü": body,
            "Hansı bazar üçün yığılıb": market, "Rəng": color, "Sürətlər qutusu": gearbox,
            "Vəziyyəti": condition, "Ötürücü": drive, "Yanacaq novu": fuel,
        }])
        row[["Sahiblər", "horsepower"]] = row[["Sahiblər", "horsepower"]].astype(float)
        price = float(model.predict(row)[0])
        low, high = (price * f for f in meta["error_band"])

        st.write("")
        left, right = st.columns([1, 1.4], gap="large")
        with left:
            st.markdown(f"""
            <div class="price-card">
              <div class="label">Təxmini bazar qiyməti</div>
              <div class="price">{fmt_azn(price)}</div>
              <div class="range">Ehtimal olunan aralıq<br><b>{fmt_azn(low)} – {fmt_azn(high)}</b></div>
            </div>""", unsafe_allow_html=True)
            st.markdown(f'<p class="footnote">{brand} {car_model}, {year} · {mileage:,} km'
                        .replace(",", " ") + "</p>", unsafe_allow_html=True)

        with right:
            history = stats[(stats["Marka"] == brand) & (stats["Model"] == car_model)]
            if len(history) >= 2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=history["production_year"], y=history["median_price"], mode="lines+markers",
                    name="Elanlarda median qiymət", line=dict(color="#9CA3AF", width=2),
                    customdata=history["listings"],
                    hovertemplate="%{x}: %{y:,.0f} AZN<br>%{customdata} elan<extra></extra>"))
                fig.add_trace(go.Scatter(
                    x=[year], y=[price], mode="markers", name="Sizin avtomobil",
                    marker=dict(color=ACCENT, size=14, line=dict(color="white", width=2)),
                    hovertemplate="Proqnoz: %{y:,.0f} AZN<extra></extra>"))
                fig.update_layout(
                    title=dict(text=f"{brand} {car_model}: illərə görə median qiymət", font=dict(size=15)),
                    height=320, margin=dict(l=10, r=10, t=50, b=10),
                    legend=dict(orientation="h", y=-0.2), yaxis_title="AZN", xaxis_title="Buraxılış ili",
                    plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            else:
                st.info("Bu model üçün qrafik qurmağa kifayət qədər elan yoxdur.")

with tab_about:
    c1, c2, c3 = st.columns(3)
    c1.metric("R² (test)", f"{metrics['r2']:.3f}")
    c2.metric("Median xəta", f"{metrics['median_abs_pct_error']:.1f}%")
    c3.metric("Öyrədildiyi elanlar", f"{meta['rows']:,}".replace(",", " "))
    st.markdown(f"""
**Necə işləyir?** Model **LightGBM** gradient boosting alqoritmidir. O, Turbo.az-dan toplanmış
açıq [Kaggle datasetindəki](https://www.kaggle.com/datasets/sehriyarmemmedli/turboaz-cars-project)
elanlar üzərində öyrədilib. Model qiyməti marka, model, il, yürüş, mühərrik, at gücü, ban növü,
vəziyyət və başqa göstəricilərə görə proqnozlaşdırır.

- Test dəstində orta mütləq xəta (MAE): **{fmt_azn(metrics['mae'])}**
- Proqnozların 80%-i həqiqi qiymətə göstərilən aralıq daxilində düşür
- USD və EUR qiymətləri 1.7 əmsalı ilə AZN-ə çevrilib

**Məhdudiyyətlər:** data müəyyən bir dövrdə toplanıb, ona görə cari bazar qiymətlərindən
fərqlənə bilər. Nadir və ya çox bahalı avtomobillər üçün xəta daha böyük ola bilər.
""")

st.divider()
st.markdown(
    '<p class="footnote">Bu, şəxsi təhsil layihəsidir və Turbo.az ilə heç bir əlaqəsi yoxdur. '
    'Nəticələr təxminidir və rəsmi qiymətləndirmə deyil. · '
    '<a href="https://github.com/mahammadhajiiyev/turbo-az-price-prediction">GitHub</a></p>',
    unsafe_allow_html=True)
