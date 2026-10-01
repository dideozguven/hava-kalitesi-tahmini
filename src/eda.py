"""Keşifsel veri analizi (EDA) - city_day.csv

İki bölümden oluşur:
  - Genel: tüm şehirler (veri kapsamı, eksik veri, şehir seçimi için)
  - Şehir: seçilen tek şehir (modellemeye yön verecek desenler)

Kullanım (proje kök klasöründen):
    python src/eda.py                          # iki bölüm, varsayılan şehir
    python src/eda.py --bolum genel            # sadece tüm şehirler
    python src/eda.py --bolum sehir --sehir Mumbai

Tablolar terminale yazdırılır, grafikler reports/figures/ altına kaydedilir:
    reports/figures/genel/     tüm şehir analizleri
    reports/figures/<sehir>/   tek şehir analizleri
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Grafikleri pencere açmadan dosyaya kaydeder
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.data import AQI_ORDER, CITY, POLLUTANTS, TARGET, load_city_day  # noqa: E402

FIG_DIR = ROOT / "reports" / "figures"
GENEL_DIR = FIG_DIR / "genel"


KAPANMA_BASLANGIC = pd.Timestamp("2020-03-25")
HINDISTAN_GUNLUK_SINIR = 60  # µg/m³, PM2.5 için 24 saatlik ulusal sınır
DIWALI = pd.to_datetime(
    ["2015-11-11", "2016-10-30", "2017-10-19", "2018-11-07", "2019-10-27"]
)
AY_ADLARI = ["Oca", "Şub", "Mar", "Nis", "May", "Haz",
             "Tem", "Ağu", "Eyl", "Eki", "Kas", "Ara"]
GUN_ADLARI = ["Pzt", "Sal", "Çar", "Per", "Cum", "Cmt", "Paz"]

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 200)


def baslik(metin: str) -> None:
    print(f"\n{'=' * 70}\n{metin}\n{'=' * 70}")


def kaydet(fig: plt.Figure, ad: str, klasor: Path = GENEL_DIR) -> None:
    klasor.mkdir(parents=True, exist_ok=True)
    yol = klasor / f"{ad}.png"
    fig.tight_layout()
    fig.savefig(yol, dpi=120)
    plt.close(fig)
    print(f"Grafik kaydedildi: {yol.relative_to(ROOT)}")


# ===========================================================================
# BÖLÜM A: TÜM ŞEHİRLER
# ===========================================================================

def genel_bakis(df: pd.DataFrame) -> None:
    baslik("A1. Genel bakış")
    print(f"Satır: {len(df):,}   Sütun: {df.shape[1]}")
    print(f"Tarih aralığı: {df['Date'].min().date()} -> {df['Date'].max().date()}")
    print(f"Şehir sayısı: {df['City'].nunique()}")
    print("\nİlk satırlar:")
    print(df.head())
    print("\nÖzet istatistikler:")
    print(df.describe().T.round(2))


def sehir_kapsami(df: pd.DataFrame) -> pd.DataFrame:
    baslik("A2. Şehir bazında veri kapsamı")
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
    kaydet(fig, "A2_sehir_olcum_donemi")
    return coverage


def eksik_veri(df: pd.DataFrame) -> None:
    baslik("A3. Eksik veri analizi")
    kolonlar = POLLUTANTS + ["AQI"]

    missing = (df[kolonlar].isna().mean() * 100).sort_values(ascending=False)
    print("Kirleticilere göre eksik oran (%):")
    print(missing.round(1))

    fig, ax = plt.subplots(figsize=(9, 4))
    missing.plot.bar(ax=ax)
    ax.set_ylabel("Eksik oran (%)")
    ax.set_title("Kirleticilere göre eksik veri oranı")
    kaydet(fig, "A3a_eksik_veri_kirletici")

    missing_by_city = df.groupby("City")[kolonlar].apply(lambda g: g.isna().mean() * 100)
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(
        missing_by_city, cmap="Reds", vmin=0, vmax=100, annot=True, fmt=".0f",
        cbar_kws={"label": "Eksik %"}, ax=ax,
    )
    ax.set_title("Şehir x kirletici eksik veri oranı (%)")
    kaydet(fig, "A3b_eksik_veri_sehir_kirletici")


def aqi_dagilimi(df: pd.DataFrame) -> None:
    baslik("A4. AQI kategorilerinin dağılımı")
    counts = df["AQI_Bucket"].value_counts().reindex(AQI_ORDER)
    print(counts)

    fig, ax = plt.subplots(figsize=(8, 4))
    counts.plot.bar(ax=ax, color=sns.color_palette("RdYlGn_r", len(AQI_ORDER)))
    ax.set_title("AQI kategorilerinin dağılımı")
    ax.set_ylabel("Gün sayısı")
    ax.tick_params(axis="x", rotation=0)
    kaydet(fig, "A4_aqi_dagilimi")


def pm25_zaman(df: pd.DataFrame, coverage: pd.DataFrame) -> None:
    baslik("A5. Zaman içinde PM2.5")
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
    kaydet(fig, "A5_pm25_aylik_trend")


def mevsimsellik_tum_sehirler(df: pd.DataFrame) -> None:
    baslik("A6. Mevsimsellik (tüm şehirler)")
    tmp = df.assign(Ay=df["Date"].dt.month, Yil=df["Date"].dt.year)
    print("Aylara göre ortalama PM2.5:")
    print(tmp.groupby("Ay")["PM2.5"].mean().round(1))
    print("Not: Şehirler veri setine farklı yıllarda katıldığı için yıllar arası "
          "fark kısmen şehir karışımının değişmesinden kaynaklanabilir.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    sns.boxplot(data=tmp, x="Ay", y="PM2.5", ax=axes[0], showfliers=False)
    axes[0].set_title("Aylara göre PM2.5 (tüm şehirler)")
    sns.boxplot(data=tmp, x="Yil", y="PM2.5", ax=axes[1], showfliers=False)
    axes[1].set_title("Yıllara göre PM2.5 (tüm şehirler)")
    kaydet(fig, "A6_mevsimsellik")


def korelasyon_tum_sehirler(df: pd.DataFrame) -> None:
    baslik("A7. Kirleticiler arası korelasyon (tüm şehirler)")
    corr = df[POLLUTANTS + ["AQI"]].corr()
    print("AQI ile korelasyon:")
    print(corr["AQI"].drop("AQI").sort_values(ascending=False).round(2))

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f", square=True, ax=ax)
    ax.set_title("Korelasyon matrisi")
    kaydet(fig, "A7_korelasyon")


def kapanma_etkisi(df: pd.DataFrame) -> None:
    baslik("A8. COVID kapanması: 25 Mart - 30 Haziran, 2019 vs 2020")

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
    kaydet(fig, "A8_kapanma_etkisi")


def genel_eda(df: pd.DataFrame) -> None:
    genel_bakis(df)
    coverage = sehir_kapsami(df)
    eksik_veri(df)
    aqi_dagilimi(df)
    pm25_zaman(df, coverage)
    mevsimsellik_tum_sehirler(df)
    korelasyon_tum_sehirler(df)
    kapanma_etkisi(df)


# ===========================================================================
# BÖLÜM B: SEÇİLEN ŞEHİR
# ===========================================================================

def sehir_verisi(df: pd.DataFrame, sehir: str) -> pd.DataFrame:
    baslik(f"B1. {sehir} için veri özeti")
    city = (
        df[df["City"] == sehir]
        .set_index("Date")
        .sort_index()
        .asfreq("D")  # Veride hiç satırı olmayan günleri NaN olarak ekler
    )
    s = city[TARGET]

    bos = s.isna()
    gruplar = (bos != bos.shift()).cumsum()
    en_uzun_bosluk = int(bos.groupby(gruplar).sum().max())

    print(f"Tarih aralığı: {city.index.min().date()} -> {city.index.max().date()}")
    print(f"Takvimdeki gün sayısı: {len(city):,}")
    print(f"{TARGET} eksik gün: {bos.sum():,} (%{100 * bos.mean():.1f})")
    print(f"En uzun ardışık eksik dönem: {en_uzun_bosluk} gün")
    print("\nKirleticilere göre eksik oran (%):")
    print((city[POLLUTANTS].isna().mean() * 100).round(1).sort_values())
    return city


def sehir_zaman_serisi(s: pd.Series, klasor: Path) -> None:
    baslik("B2. Günlük PM2.5 zaman serisi")
    fig, ax = plt.subplots(figsize=(14, 4))
    s.plot(ax=ax, lw=0.6, alpha=0.6, label="Günlük")
    s.rolling(30, min_periods=15).mean().plot(ax=ax, lw=2, label="30 günlük ortalama")
    ax.axhline(HINDISTAN_GUNLUK_SINIR, color="red", ls="--", lw=1, label="Günlük sınır")
    ax.axvspan(KAPANMA_BASLANGIC, s.index.max(), color="grey", alpha=0.2, label="Kapanma")
    ax.set_ylabel(f"{TARGET} (µg/m³)")
    ax.set_title(f"Günlük {TARGET}")
    ax.legend(loc="upper right")
    kaydet(fig, "B2_zaman_serisi", klasor)


def sehir_mevsimsellik(s: pd.Series, klasor: Path) -> None:
    baslik("B3. Mevsimsellik (seçilen şehir)")
    tmp = s.to_frame(TARGET).assign(Ay=s.index.month, Yil=s.index.year)

    aylik = tmp.groupby("Ay")[TARGET].mean()
    aylik.index = AY_ADLARI
    print("Aylara göre ortalama PM2.5:")
    print(aylik.round(1))
    print(f"\nEn kirli ay / en temiz ay oranı: {aylik.max() / aylik.min():.1f}x")

    pivot = tmp.pivot_table(index="Yil", columns="Ay", values=TARGET, aggfunc="mean")
    pivot.columns = [AY_ADLARI[a - 1] for a in pivot.columns]

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
    sns.boxplot(data=tmp, x="Ay", y=TARGET, ax=axes[0], showfliers=False)
    axes[0].set_xticks(range(12))
    axes[0].set_xticklabels(AY_ADLARI)
    axes[0].set_title("Aylara göre dağılım")
    sns.heatmap(pivot, cmap="Reds", annot=True, fmt=".0f", ax=axes[1],
                cbar_kws={"label": f"{TARGET} (µg/m³)"})
    axes[1].set_title("Yıl x ay ortalaması")
    kaydet(fig, "B3_mevsimsellik", klasor)


def sehir_haftalik_desen(s: pd.Series, klasor: Path) -> None:
    baslik("B4. Haftanın günlerine göre PM2.5")
    haftalik = s.groupby(s.index.dayofweek).mean()
    haftalik.index = GUN_ADLARI
    print(haftalik.round(1))
    fark = 100 * (haftalik.max() - haftalik.min()) / haftalik.mean()
    print(f"\nEn yüksek ve en düşük gün arasındaki fark: ortalamanın %{fark:.1f}'i")

    fig, ax = plt.subplots(figsize=(7, 4))
    haftalik.plot.bar(ax=ax)
    ax.set_ylabel(f"Ortalama {TARGET}")
    ax.tick_params(axis="x", rotation=0)
    ax.set_title("Haftanın günlerine göre ortalama")
    kaydet(fig, "B4_haftalik_desen", klasor)


def sehir_korelasyon(city: pd.DataFrame, klasor: Path) -> None:
    baslik("B5. PM2.5 ile diğer kirleticilerin korelasyonu (seçilen şehir)")
    digerleri = [p for p in POLLUTANTS if p != TARGET and city[p].notna().mean() > 0.5]

    tablo = pd.DataFrame({
        "ayni_gun": city[digerleri].corrwith(city[TARGET]),
        "bir_gun_once": city[digerleri].shift(1).corrwith(city[TARGET]),
    }).sort_values("bir_gun_once", ascending=False)
    print("(Yarısından fazlası boş olan kirleticiler dışarıda bırakıldı)")
    print(tablo.round(2))

    fig, ax = plt.subplots(figsize=(9, 5))
    tablo.plot.barh(ax=ax)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel(f"{TARGET} ile korelasyon")
    ax.set_title("Aynı gün ve bir gün önceki değerlerle korelasyon")
    kaydet(fig, "B5_korelasyon", klasor)


def sehir_otokorelasyon(s: pd.Series, klasor: Path, max_lag: int = 30) -> None:
    baslik("B6. PM2.5 otokorelasyonu")
    acf = pd.Series({lag: s.autocorr(lag) for lag in range(1, max_lag + 1)})
    print("İlk 7 gecikme:")
    print(acf.head(7).round(2))

    fig, ax = plt.subplots(figsize=(11, 4))
    acf.plot.bar(ax=ax)
    ax.set_xlabel("Gecikme (gün)")
    ax.set_ylabel("Korelasyon")
    ax.set_title(f"{TARGET}: bugünkü değerin n gün önceki değerle korelasyonu")
    kaydet(fig, "B6_otokorelasyon", klasor)


def sehir_diwali_etkisi(s: pd.Series, klasor: Path, pencere: int = 10) -> None:
    baslik("B7. Diwali çevresinde PM2.5")
    satirlar = {}
    for gun in DIWALI:
        aralik = pd.date_range(gun - pd.Timedelta(days=pencere),
                               gun + pd.Timedelta(days=pencere))
        degerler = s.reindex(aralik)
        if degerler.notna().sum() < pencere:
            continue
        satirlar[gun.year] = pd.Series(degerler.values, index=range(-pencere, pencere + 1))

    if not satirlar:
        print("Bu şehir için Diwali dönemlerinde yeterli veri yok.")
        return

    tablo = pd.DataFrame(satirlar)
    oncesi = tablo.loc[-pencere:-3].mean()
    sonrasi = tablo.loc[0:1].mean()
    ozet = pd.DataFrame({"oncesi_ort": oncesi, "diwali_ve_ertesi": sonrasi})
    ozet["artis_yuzde"] = 100 * (ozet["diwali_ve_ertesi"] - ozet["oncesi_ort"]) / ozet["oncesi_ort"]
    print(ozet.round(1))

    fig, ax = plt.subplots(figsize=(10, 4))
    tablo.plot(ax=ax, alpha=0.5)
    tablo.mean(axis=1).plot(ax=ax, color="black", lw=2.5, label="Ortalama")
    ax.axvline(0, color="red", ls="--", lw=1)
    ax.set_xlabel("Diwali'ye göre gün")
    ax.set_ylabel(TARGET)
    ax.set_title("Diwali çevresinde PM2.5 (yıllara göre)")
    ax.legend(ncol=3)
    kaydet(fig, "B7_diwali", klasor)


def sehir_dagilim(s: pd.Series, klasor: Path) -> None:
    baslik("B8. PM2.5 dağılımı")
    temiz = s.dropna()
    print(temiz.describe().round(1))
    print(f"\nÇarpıklık (skewness): {temiz.skew():.2f}")
    asim = (temiz > HINDISTAN_GUNLUK_SINIR).mean() * 100
    print(f"Günlük sınırın ({HINDISTAN_GUNLUK_SINIR} µg/m³) aşıldığı günler: %{asim:.1f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    sns.histplot(temiz, bins=60, ax=axes[0])
    axes[0].axvline(HINDISTAN_GUNLUK_SINIR, color="red", ls="--")
    axes[0].set_title(f"{TARGET} dağılımı")
    sns.histplot(temiz.clip(lower=1).apply("log"), bins=60, ax=axes[1])
    axes[1].set_title(f"log({TARGET}) dağılımı")
    kaydet(fig, "B8_dagilim", klasor)


def sehir_eda(df: pd.DataFrame, sehir: str) -> None:
    klasor = FIG_DIR / sehir.lower().replace(" ", "_")
    city = sehir_verisi(df, sehir)
    s = city[TARGET]
    sehir_zaman_serisi(s, klasor)
    sehir_mevsimsellik(s, klasor)
    sehir_haftalik_desen(s, klasor)
    sehir_korelasyon(city, klasor)
    sehir_otokorelasyon(s, klasor)
    sehir_diwali_etkisi(s, klasor)
    sehir_dagilim(s, klasor)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hava kalitesi EDA")
    parser.add_argument("--bolum", choices=["genel", "sehir", "hepsi"], default="hepsi")
    parser.add_argument("--sehir", default=CITY)
    args = parser.parse_args()

    df = load_city_day()

    if args.bolum in ("genel", "hepsi"):
        genel_eda(df)

    if args.bolum in ("sehir", "hepsi"):
        if args.sehir not in set(df["City"]):
            print(f"'{args.sehir}' bulunamadı. Mevcut şehirler:")
            print(", ".join(sorted(df["City"].unique())))
            sys.exit(1)
        sehir_eda(df, args.sehir)

    baslik("Bitti")
    print(f"Tüm grafikler: {FIG_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
