"""Keşifsel veri analizi (EDA) - city_day.csv

Kullanım (proje kök klasöründen):
    python scripts/eda.py

Tablolar terminale yazdırılır, grafikler reports/figures/ altına kaydedilir.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Grafikleri pencere açmadan dosyaya kaydeder
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.data import AQI_ORDER, POLLUTANTS, load_city_day  # noqa: E402

FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 200)


def baslik(metin: str) -> None:
    print(f"\n{'=' * 70}\n{metin}\n{'=' * 70}")


def kaydet(fig: plt.Figure, ad: str) -> None:
    yol = FIG_DIR / f"{ad}.png"
    fig.tight_layout()
    fig.savefig(yol, dpi=120)
    plt.close(fig)
    print(f"Grafik kaydedildi: {yol.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# 1. Genel bakış
# ---------------------------------------------------------------------------
def genel_bakis(df: pd.DataFrame) -> None:
    baslik("1. Genel bakış")
    print(f"Satır: {len(df):,}   Sütun: {df.shape[1]}")
    print(f"Tarih aralığı: {df['Date'].min().date()} -> {df['Date'].max().date()}")
    print(f"Şehir sayısı: {df['City'].nunique()}")
    print("\nİlk satırlar:")
    print(df.head())
    print("\nÖzet istatistikler:")
    print(df.describe().T.round(2))


# ---------------------------------------------------------------------------
# 2. Şehir bazında veri kapsamı
# ---------------------------------------------------------------------------
def sehir_kapsami(df: pd.DataFrame) -> pd.DataFrame:
    baslik("2. Şehir bazında veri kapsamı")
    coverage = (
        df.groupby("City")
        .agg(
            ilk_tarih=("Date", "min"),
            son_tarih=("Date", "max"),
            gun_sayisi=("Date", "count"),
            pm25_dolu=("PM2.5", "count"),
        )
        .assign(pm25_doluluk_yuzde=lambda x: (100 * x["pm25_dolu"] / x["gun_sayisi"]).round(1))
        .sort_values("gun_sayisi", ascending=False)
    )
    print(coverage)

    sirali = coverage.sort_values("ilk_tarih")
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, (_, row) in enumerate(sirali.iterrows()):
        ax.plot([row["ilk_tarih"], row["son_tarih"]], [i, i], lw=6)
    ax.set_yticks(range(len(sirali)))
    ax.set_yticklabels(sirali.index)
    ax.set_title("Şehirlere göre ölçüm dönemi")
    kaydet(fig, "02_sehir_olcum_donemi")
    return coverage


# ---------------------------------------------------------------------------
# 3. Eksik veri analizi
# ---------------------------------------------------------------------------
def eksik_veri(df: pd.DataFrame) -> None:
    baslik("3. Eksik veri analizi")
    kolonlar = POLLUTANTS + ["AQI"]

    missing = (df[kolonlar].isna().mean() * 100).sort_values(ascending=False)
    print("Kirleticilere göre eksik oran (%):")
    print(missing.round(1))

    fig, ax = plt.subplots(figsize=(9, 4))
    missing.plot.bar(ax=ax)
    ax.set_ylabel("Eksik oran (%)")
    ax.set_title("Kirleticilere göre eksik veri oranı")
    kaydet(fig, "03a_eksik_veri_kirletici")

    missing_by_city = df.groupby("City")[kolonlar].apply(lambda g: g.isna().mean() * 100)
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(
        missing_by_city, cmap="Reds", vmin=0, vmax=100, annot=True, fmt=".0f",
        cbar_kws={"label": "Eksik %"}, ax=ax,
    )
    ax.set_title("Şehir x kirletici eksik veri oranı (%)")
    kaydet(fig, "03b_eksik_veri_sehir_kirletici")


# ---------------------------------------------------------------------------
# 4. AQI kategorileri
# ---------------------------------------------------------------------------
def aqi_dagilimi(df: pd.DataFrame) -> None:
    baslik("4. AQI kategorilerinin dağılımı")
    counts = df["AQI_Bucket"].value_counts().reindex(AQI_ORDER)
    print(counts)

    fig, ax = plt.subplots(figsize=(8, 4))
    counts.plot.bar(ax=ax, color=sns.color_palette("RdYlGn_r", len(AQI_ORDER)))
    ax.set_title("AQI kategorilerinin dağılımı")
    ax.set_ylabel("Gün sayısı")
    ax.tick_params(axis="x", rotation=0)
    kaydet(fig, "04_aqi_dagilimi")


# ---------------------------------------------------------------------------
# 5. Zaman içinde PM2.5
# ---------------------------------------------------------------------------
def pm25_zaman(df: pd.DataFrame, coverage: pd.DataFrame) -> None:
    baslik("5. Zaman içinde PM2.5")
    top_cities = coverage.sort_values("pm25_dolu", ascending=False).head(6).index
    print("En çok PM2.5 verisi olan 6 şehir:", ", ".join(top_cities))

    monthly = (
        df[df["City"].isin(top_cities)]
        .set_index("Date")
        .groupby("City")["PM2.5"]
        .resample("MS")
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(13, 5))
    sns.lineplot(data=monthly, x="Date", y="PM2.5", hue="City", ax=ax)
    ax.set_title("En çok PM2.5 verisi olan 6 şehirde aylık ortalama PM2.5")
    kaydet(fig, "05_pm25_aylik_trend")


# ---------------------------------------------------------------------------
# 6. Mevsimsellik
# ---------------------------------------------------------------------------
def mevsimsellik(df: pd.DataFrame) -> None:
    baslik("6. Mevsimsellik")
    tmp = df.assign(Ay=df["Date"].dt.month, Yil=df["Date"].dt.year)
    print("Aylara göre ortalama PM2.5:")
    print(tmp.groupby("Ay")["PM2.5"].mean().round(1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    sns.boxplot(data=tmp, x="Ay", y="PM2.5", ax=axes[0], showfliers=False)
    axes[0].set_title("Aylara göre PM2.5 (tüm şehirler)")
    sns.boxplot(data=tmp, x="Yil", y="PM2.5", ax=axes[1], showfliers=False)
    axes[1].set_title("Yıllara göre PM2.5 (tüm şehirler)")
    kaydet(fig, "06_mevsimsellik")


# ---------------------------------------------------------------------------
# 7. Korelasyon
# ---------------------------------------------------------------------------
def korelasyon(df: pd.DataFrame) -> None:
    baslik("7. Kirleticiler arası korelasyon")
    corr = df[POLLUTANTS + ["AQI"]].corr()
    print("AQI ile korelasyon:")
    print(corr["AQI"].drop("AQI").sort_values(ascending=False).round(2))

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f", square=True, ax=ax)
    ax.set_title("Korelasyon matrisi")
    kaydet(fig, "07_korelasyon")


# ---------------------------------------------------------------------------
# 8. COVID kapanması
# ---------------------------------------------------------------------------
def kapanma_etkisi(df: pd.DataFrame) -> None:
    baslik("8. COVID kapanması: 25 Mart - 30 Haziran, 2019 vs 2020")

    def donem(yil: int) -> pd.Series:
        mask = (df["Date"] >= f"{yil}-03-25") & (df["Date"] <= f"{yil}-06-30")
        return df[mask].groupby("City")["PM2.5"].mean()

    lockdown = pd.DataFrame({"2019": donem(2019), "2020": donem(2020)}).dropna()
    lockdown["degisim_yuzde"] = (
        100 * (lockdown["2020"] - lockdown["2019"]) / lockdown["2019"]
    ).round(1)
    lockdown = lockdown.sort_values("degisim_yuzde")
    print(lockdown.round(1))

    fig, ax = plt.subplots(figsize=(9, 6))
    lockdown["degisim_yuzde"].plot.barh(ax=ax)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("PM2.5 değişimi (%)")
    ax.set_title("25 Mart - 30 Haziran: 2020'nin 2019'a göre PM2.5 değişimi")
    kaydet(fig, "08_kapanma_etkisi")


def main() -> None:
    df = load_city_day()
    genel_bakis(df)
    coverage = sehir_kapsami(df)
    eksik_veri(df)
    aqi_dagilimi(df)
    pm25_zaman(df, coverage)
    mevsimsellik(df)
    korelasyon(df)
    kapanma_etkisi(df)
    baslik("Bitti")
    print(f"Tüm grafikler: {FIG_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()