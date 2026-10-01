"""Delhi için yarının PM2.5 değerini SARIMAX ile tahmin eder.

LightGBM ile adil karşılaştırma için aynı temiz veri, aynı bölme, aynı
değerlendirme günleri ve aynı metrikler kullanılır (src.train_lightgbm'den).

Model:
    - Hedef: log(1 + PM2.5)   (dağılım sağa çarpık, log daha simetrik)
    - AR/MA: son günlerin değerleri ve tahmin hataları
    - Mevsimsel kısım (s=7): haftalık desen
    - Dış değişkenler (X): yıllık mevsimsellik için Fourier terimleri
      (sin/cos). Tarihe bağlı oldukları için yarın için de bilinirler.

Değerlendirme: Her gün için sadece bir önceki güne kadarki gerçek değerler
kullanılarak "yarın" tahmin edilir (kayan bir günlük tahmin).

Kullanım (proje kök klasöründen):
    python -m src.train_sarimax            # 2015-2018'de eğit, 2019'da ölç
    python -m src.train_sarimax --final    # 2015-2019'da eğit, 2020'de ölç

Test setine (--final) sadece tüm denemeler bittikten sonra bir kez bakılmalı.
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.data import CITY, ROOT, TARGET
from src.preprocess import FILLED_FLAG
from src.train_lightgbm import TRAIN_END, VAL_END, build_features, load_clean, metrics, split

FIGURE_DIR = ROOT / "reports" / "figures"

ORDER = (2, 0, 1)               # (p, d, q): 2 gün geriye bak, fark alma yok, 1 hata terimi
SEASONAL_ORDER = (1, 0, 1, 7)   # (P, D, Q, s): haftalık mevsimsellik
FOURIER_K = 2                   # yıllık mevsimsellik için kaç sin/cos çifti


# ---------------------------------------------------------------------------
# Veri hazırlığı
# ---------------------------------------------------------------------------
def prepare_series(df: pd.DataFrame) -> pd.Series:
    """Hedef seriyi hazırlar.

    - Interpolasyonla doldurulmuş PM2.5 günleri tekrar NaN yapılır. SARIMAX
      eksik günleri kendisi yönetebildiği için uydurma değerlere gerek yok;
      böylece interpolasyonun gelecekteki değeri kullanmasından doğan sızıntı
      da tamamen ortadan kalkar.
    - log1p dönüşümü uygulanır.
    """
    y = df[TARGET].astype(float).copy()
    doldurulan = df[FILLED_FLAG].fillna(0).astype(bool)
    y[doldurulan] = np.nan
    return np.log1p(y)


def fourier_terms(index: pd.DatetimeIndex, k: int = FOURIER_K) -> pd.DataFrame:
    """Yılın gününden sin/cos terimleri üretir (yıllık mevsimsellik)."""
    t = index.dayofyear / 365.25
    cols = {}
    for i in range(1, k + 1):
        cols[f"sin{i}"] = np.sin(2 * np.pi * i * t)
        cols[f"cos{i}"] = np.cos(2 * np.pi * i * t)
    return pd.DataFrame(cols, index=index)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def fit(y: pd.Series, X: pd.DataFrame, end: str):
    """Modeli verilen tarihe kadarki veriyle eğitir."""
    model = SARIMAX(
        y[:end], exog=X[:end],
        order=ORDER, seasonal_order=SEASONAL_ORDER, trend="c",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # statsmodels'in yakınsama uyarılarını gizle
        res = model.fit(disp=False, maxiter=200)
    print(f"\nSARIMAX{ORDER}x{SEASONAL_ORDER} eğitildi  (AIC: {res.aic:.1f})")
    print("Katsayılar:")
    print(res.params.round(3).to_string())
    return res


def one_step_predictions(res, y: pd.Series, X: pd.DataFrame, end, dates: pd.DatetimeIndex):
    """Her tarih için, bir önceki güne kadarki gerçek veriyle yapılan tahmin.

    res.apply: Eğitilmiş katsayıları değiştirmeden modeli daha uzun veriye uygular.
    get_prediction (dynamic=False): Her gün için sadece o güne kadarki gözlemleri
    kullanarak bir adım ileri tahmin üretir.
    """
    full = res.apply(y[:end], exog=X[:end])
    pred = full.get_prediction(start=dates.min(), end=dates.max())
    mean = np.expm1(pred.predicted_mean).reindex(dates)
    ci = np.expm1(pred.conf_int(alpha=0.05)).reindex(dates)
    ci.columns = ["alt", "ust"]
    return mean, ci


# ---------------------------------------------------------------------------
# Raporlama
# ---------------------------------------------------------------------------
def report(name: str, df: pd.DataFrame, pred: pd.Series, ci: pd.DataFrame) -> None:
    gercek = df["target"].to_numpy()
    tahmin = pred.to_numpy()
    baseline = df["pm25_lag1"]
    mask = baseline.notna().to_numpy()

    rows = {
        "Persistence (yarın = bugün)": metrics(gercek[mask], baseline.to_numpy()[mask]),
        "SARIMAX": metrics(gercek[mask], tahmin[mask]),
    }
    print(f"\n=== {name} ({mask.sum()} gün) ===")
    print(pd.DataFrame(rows).T.round(2).to_string())

    hata = pd.Series(np.abs(gercek - tahmin), index=pred.index)
    print("\nAylara göre SARIMAX MAE:")
    print(hata.groupby(hata.index.month).mean().round(1).to_string())

    icinde = (gercek >= ci["alt"].to_numpy()) & (gercek <= ci["ust"].to_numpy())
    genislik = (ci["ust"] - ci["alt"]).median()
    print(f"\n%95 güven aralığı: gerçek değerlerin %{100 * icinde.mean():.1f}'i aralığın içinde "
          f"(ideal: ~%95), ortanca aralık genişliği {genislik:.0f} µg/m³")


def plot_predictions(name: str, pred: pd.Series, ci: pd.DataFrame, df: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(ci.index, ci["alt"], ci["ust"], alpha=0.2, label="%95 güven aralığı")
    ax.plot(df["target_date"], df["target"], label="Gerçek", lw=1)
    ax.plot(pred.index, pred, label="SARIMAX", lw=1)
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_title(f"{CITY} yarının PM2.5 tahmini (SARIMAX) - {name}")
    ax.legend()
    fig.tight_layout()
    path = FIGURE_DIR / f"sarimax_{name.lower()}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Grafik kaydedildi: {path}")


# ---------------------------------------------------------------------------
# Ana akış
# ---------------------------------------------------------------------------
def main(final: bool = False) -> None:
    df = load_clean()
    train, val, test = split(build_features(df))  # sadece değerlendirme günleri için

    y = prepare_series(df)
    X = fourier_terms(df.index)

    if final:
        res = fit(y, X, end=VAL_END)            # 2015-2019
        eval_df, son, ad = test, df.index.max(), "test"
    else:
        res = fit(y, X, end=TRAIN_END)          # 2015-2018
        eval_df, son, ad = val, VAL_END, "val"

    dates = pd.DatetimeIndex(eval_df["target_date"])
    pred, ci = one_step_predictions(res, y, X, son, dates)
    report("Test 2020" if final else "Doğrulama 2019", eval_df, pred, ci)
    plot_predictions(ad, pred, ci, eval_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true",
                        help="2015-2019'da eğit ve 2020 test setinde değerlendir")
    main(final=parser.parse_args().final)