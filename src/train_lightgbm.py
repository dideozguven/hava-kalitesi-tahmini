"""Delhi için yarının PM2.5 değerini LightGBM ile tahmin eder.

Akış:
    temiz veri -> özellikler -> zamansal bölme -> baseline -> LightGBM -> değerlendirme

Kullanım (proje kök klasöründen):
    python -m src.train_lightgbm            # train'de eğit, 2019 doğrulamada ölç
    python -m src.train_lightgbm --final    # train+val'de yeniden eğit, 2020 testinde ölç

Test setine (--final) sadece tüm denemeler bittikten sonra bir kez bakılmalı.
"""
import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data import CITY, ROOT, TARGET
from src.preprocess import FILLED_FLAG, MAX_GAP, PROCESSED_DIR, run_pipeline

MODEL_DIR = ROOT / "models"
FIGURE_DIR = ROOT / "reports" / "figures"

TRAIN_END = "2018-12-31"
VAL_END = "2019-12-31"

TARGET_LAGS = [1, 2, 3, 7, 14]   # hedef günden kaç gün önceki PM2.5 değerleri
ROLL_WINDOWS = [3, 7, 14]

PARAMS = {
    "objective": "regression",
    "metric": ["l1", "l2"],
    "learning_rate": 0.03,
    "num_leaves": 15,            # veri küçük (~2000 gün), ağaçları sade tut
    "min_data_in_leaf": 30,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "seed": 42,
    "verbose": -1,
}
NUM_BOOST_ROUND = 3000
EARLY_STOPPING = 100


def load_clean(city: str = CITY) -> pd.DataFrame:
    """Temiz veriyi okur; dosya yoksa ön işlemeyi çalıştırır."""
    path = PROCESSED_DIR / f"{city.lower()}_clean.csv"
    if not path.exists():
        print(f"{path.name} bulunamadı, ön işleme çalıştırılıyor...")
        run_pipeline(city)
    return pd.read_csv(path, parse_dates=["Date"], index_col="Date").asfreq("D")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Her satır = bugün (t). Hedef = yarının (t+1) PM2.5 değeri.

    Tüm özellikler sadece t ve öncesindeki bilgiyi kullanır.
    """
    pm = df[TARGET]
    feats = pd.DataFrame(index=df.index)

    # Hedefin geçmiş değerleri (lag1 = bugün, lag2 = dün, ...)
    for lag in TARGET_LAGS:
        feats[f"pm25_lag{lag}"] = pm.shift(lag - 1)

    # Hareketli istatistikler (bugün dahil, yarın hariç)
    for w in ROLL_WINDOWS:
        feats[f"pm25_ort{w}"] = pm.rolling(w, min_periods=max(2, w // 2)).mean()
    feats["pm25_std7"] = pm.rolling(7, min_periods=4).std()
    feats["pm25_degisim"] = pm - pm.shift(1)

    # Diğer kirleticilerin bugünkü değerleri
    diger = [c for c in df.columns if c not in (TARGET, FILLED_FLAG)]
    for col in diger:
        feats[f"{col}_lag1"] = df[col]

    # Hedef günün takvim bilgisi
    hedef_gun = df.index + pd.Timedelta(days=1)
    feats["ay"] = hedef_gun.month
    feats["haftanin_gunu"] = hedef_gun.dayofweek
    feats["yil_sin"] = np.sin(2 * np.pi * hedef_gun.dayofyear / 365.25)
    feats["yil_cos"] = np.cos(2 * np.pi * hedef_gun.dayofyear / 365.25)

    feats["target"] = pm.shift(-1)
    feats["target_date"] = hedef_gun

    # Sızıntı koruması: interpolasyonla doldurulan bir gün, en fazla MAX_GAP gün
    # sonrasındaki gerçek değeri kullanır. Hedef günün kendisi veya hedefe bu kadar
    # yakın doldurulmuş PM2.5 değerleri olan satırlar eğitimden ve ölçümden çıkarılır.
    flag = df[FILLED_FLAG].fillna(0).astype(bool)
    riskli = flag.shift(-1, fill_value=False)
    for k in range(MAX_GAP):
        riskli |= flag.shift(k, fill_value=False)
    feats["riskli"] = riskli

    return feats


def split(feats: pd.DataFrame):
    """Hedef tarihine göre zamansal bölme; hedefi boş veya riskli satırları atar."""
    kullanilabilir = feats[feats["target"].notna() & ~feats["riskli"]]
    atilan = len(feats) - len(kullanilabilir)
    print(f"Kullanılabilir satır: {len(kullanilabilir)} "
          f"(hedefi boş veya doldurulmuş {atilan} satır atıldı)")

    t = kullanilabilir["target_date"]
    train = kullanilabilir[t <= TRAIN_END]
    val = kullanilabilir[(t > TRAIN_END) & (t <= VAL_END)]
    test = kullanilabilir[t > VAL_END]
    print(f"Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")
    return train, val, test


def feature_columns(feats: pd.DataFrame) -> list[str]:
    return [c for c in feats.columns if c not in ("target", "target_date", "riskli")]


def metrics(y_true, y_pred) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def report(name: str, df: pd.DataFrame, pred: np.ndarray) -> None:
    """Baseline ve model skorlarını tablo halinde yazdırır."""
    baseline = df["pm25_lag1"]
    mask = baseline.notna()  # baseline için bugünün değeri gerekli
    rows = {
        "Persistence (yarın = bugün)": metrics(df["target"][mask], baseline[mask]),
        "LightGBM": metrics(df["target"][mask], pred[mask.to_numpy()]),
    }
    print(f"\n=== {name} ({mask.sum()} gün) ===")
    print(pd.DataFrame(rows).T.round(2).to_string())

    hata = pd.Series(np.abs(df["target"].to_numpy() - pred), index=df["target_date"])
    aylik = hata.groupby(hata.index.month).mean().round(1)
    print("\nAylara göre LightGBM MAE:")
    print(aylik.to_string())


def plot_predictions(name: str, df: pd.DataFrame, pred: np.ndarray) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["target_date"], df["target"], label="Gerçek", lw=1)
    ax.plot(df["target_date"], pred, label="LightGBM", lw=1)
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_title(f"{CITY} yarının PM2.5 tahmini - {name}")
    ax.legend()
    fig.tight_layout()
    path = FIGURE_DIR / f"lightgbm_{name.lower()}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Grafik kaydedildi: {path}")


def print_importance(model: lgb.Booster, top: int = 10) -> None:
    imp = pd.Series(model.feature_importance("gain"), index=model.feature_name())
    imp = (imp / imp.sum() * 100).sort_values(ascending=False).head(top)
    print(f"\nEn önemli {top} özellik (gain, %):")
    print(imp.round(1).to_string())


def main(final: bool = False) -> None:
    feats = build_features(load_clean())
    cols = feature_columns(feats)
    train, val, test = split(feats)

    dtrain = lgb.Dataset(train[cols], train["target"])
    dval = lgb.Dataset(val[cols], val["target"], reference=dtrain)
    model = lgb.train(
        PARAMS, dtrain, num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dtrain, dval], valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False),
                   lgb.log_evaluation(0)],
    )
    best_iter = model.best_iteration
    print(f"\nEn iyi iterasyon: {best_iter}")

    if not final:
        pred = model.predict(val[cols], num_iteration=best_iter)
        report("Doğrulama 2019", val, pred)
        plot_predictions("val", val, pred)
        print_importance(model)
        return

    # Final: train+val'de, doğrulamada bulunan iterasyon sayısıyla yeniden eğit
    full = pd.concat([train, val])
    final_model = lgb.train(PARAMS, lgb.Dataset(full[cols], full["target"]),
                            num_boost_round=best_iter)
    pred = final_model.predict(test[cols])
    report("Test 2020", test, pred)
    plot_predictions("test", test, pred)
    print_importance(final_model)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"lightgbm_{CITY.lower()}.txt"
    final_model.save_model(str(path))
    print(f"\nModel kaydedildi: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true",
                        help="train+val'de yeniden eğit ve test setinde değerlendir")
    main(final=parser.parse_args().final)