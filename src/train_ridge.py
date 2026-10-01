"""Delhi için yarının PM2.5 değerini Ridge regresyon ile tahmin eder ve
LightGBM ile aynı veri, aynı özellikler ve aynı bölme üzerinde karşılaştırır.

Amaç: "LightGBM'in doğrusal olmayan yapısı gerçekten fayda sağlıyor mu?"
sorusunu cevaplamak. Ridge, özelliklerin ağırlıklı toplamıyla tahmin yapan
doğrusal bir modeldir; aşırı büyük katsayıları alpha ile cezalandırır.

Kullanım (proje kök klasöründen):
    python -m src.train_ridge            # train'de eğit, 2019 doğrulamada karşılaştır
    python -m src.train_ridge --final    # train+val'de yeniden eğit, 2020 testinde ölç

Test setine (--final) sadece tüm denemeler bittikten sonra bir kez bakılmalı.
"""
import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.train_lightgbm import (EARLY_STOPPING, NUM_BOOST_ROUND, PARAMS,
                                build_features, feature_columns, load_clean,
                                metrics, split)

ALPHAS = [0.01, 0.1, 1, 3, 10, 30, 100, 300, 1000]


def ridge_features(feats: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Doğrusal model için takvim özelliklerini uyarlar.

    LightGBM "ay = 11" bilgisini eşik olarak kullanabilir, ama doğrusal bir model
    ay numarasını büyüklük olarak okur (Aralık > Ocak gibi). Bu yüzden:
    - ay çıkarılır; mevsim bilgisini yil_sin ve yil_cos zaten taşıyor.
    - haftanin_gunu one-hot sütunlara çevrilir.
    """
    X = feats[cols].drop(columns=["ay", "haftanin_gunu"])
    gunler = pd.get_dummies(feats["haftanin_gunu"], prefix="gun", dtype=float)
    gunler = gunler.reindex(columns=[f"gun_{i}" for i in range(7)], fill_value=0.0)
    return pd.concat([X, gunler], axis=1)


def make_ridge(alpha: float):
    """Eksik değer doldurma + ölçekleme + Ridge.

    LightGBM eksik değerlerle kendi başa çıkar, Ridge çıkamaz. Doldurma ve
    ölçekleme parametreleri pipeline içinde sadece eğitim verisinden öğrenilir;
    böylece doğrulama/test verisinden bilgi sızmaz.
    """
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=alpha),
    )


def choose_alpha(X_train, y_train, X_val, y_val) -> float:
    print("\nAlpha seçimi (doğrulama MAE):")
    sonuclar = {}
    for a in ALPHAS:
        pred = make_ridge(a).fit(X_train, y_train).predict(X_val)
        sonuclar[a] = metrics(y_val, pred)["MAE"]
        print(f"  alpha={a:<7} MAE={sonuclar[a]:.2f}")
    best = min(sonuclar, key=sonuclar.get)
    print(f"Seçilen alpha: {best}")
    return best


def train_lightgbm(train, val, cols):
    """Karşılaştırma için LightGBM'i eğitim scriptindeki ayarlarla eğitir."""
    dtrain = lgb.Dataset(train[cols], train["target"])
    dval = lgb.Dataset(val[cols], val["target"], reference=dtrain)
    return lgb.train(
        PARAMS, dtrain, num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dval], valid_names=["val"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False),
                   lgb.log_evaluation(0)],
    )


def compare(name: str, df: pd.DataFrame, preds: dict) -> None:
    """Baseline, Ridge ve LightGBM skorlarını ve aylık MAE'leri yan yana yazdırır."""
    mask = df["pm25_lag1"].notna().to_numpy()
    y = df["target"].to_numpy()[mask]
    tum = {"Persistence (yarın = bugün)": df["pm25_lag1"].to_numpy()[mask]}
    tum.update({k: v[mask] for k, v in preds.items()})

    print(f"\n=== {name} ({mask.sum()} gün) ===")
    print(pd.DataFrame({k: metrics(y, p) for k, p in tum.items()}).T.round(2).to_string())

    ay = pd.to_datetime(df["target_date"]).dt.month.to_numpy()[mask]
    aylik = pd.DataFrame({k: np.abs(y - p) for k, p in tum.items()})
    aylik["ay"] = ay
    print("\nAylara göre MAE:")
    print(aylik.groupby("ay").mean().round(1).to_string())


def print_coefficients(model, columns, top: int = 12) -> None:
    """Ölçeklenmiş özellikler üzerindeki katsayılar.

    Özellikler standartlaştırıldığı için katsayılar birbiriyle karşılaştırılabilir:
    "bu özellik 1 standart sapma artınca tahmin kaç µg/m³ değişiyor".
    """
    coef = pd.Series(model[-1].coef_, index=columns)
    sirali = coef.reindex(coef.abs().sort_values(ascending=False).index).head(top)
    print(f"\nEn büyük {top} Ridge katsayısı (µg/m³ / standart sapma):")
    print(sirali.round(2).to_string())


def main(final: bool = False) -> None:
    feats = build_features(load_clean())
    cols = feature_columns(feats)
    train, val, test = split(feats)

    Xr_train, Xr_val, Xr_test = (ridge_features(d, cols) for d in (train, val, test))
    alpha = choose_alpha(Xr_train, train["target"], Xr_val, val["target"])
    lgbm = train_lightgbm(train, val, cols)

    if not final:
        ridge = make_ridge(alpha).fit(Xr_train, train["target"])
        compare("Doğrulama 2019", val, {
            "Ridge": ridge.predict(Xr_val),
            "LightGBM": lgbm.predict(val[cols], num_iteration=lgbm.best_iteration),
        })
        print_coefficients(ridge, Xr_train.columns)
        return

    full = pd.concat([train, val])
    Xr_full = pd.concat([Xr_train, Xr_val])
    ridge = make_ridge(alpha).fit(Xr_full, full["target"])
    lgbm_final = lgb.train(PARAMS, lgb.Dataset(full[cols], full["target"]),
                           num_boost_round=lgbm.best_iteration)
    compare("Test 2020", test, {
        "Ridge": ridge.predict(Xr_test),
        "LightGBM": lgbm_final.predict(test[cols]),
    })
    print_coefficients(ridge, Xr_full.columns)


if __name__ == "__main__":
    pd.set_option("display.width", 120)
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true",
                        help="train+val'de yeniden eğit ve test setinde değerlendir")
    main(final=parser.parse_args().final)