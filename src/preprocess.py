"""Veri ön işleme: city_day.csv'den seçilen şehir için temiz bir günlük seri üretir.

Akış (run_pipeline):
    load_city_day -> filter_city -> complete_dates -> clean_invalid
                  -> select_columns -> fill_short_gaps -> save_processed

Not: clean_invalid, select_columns'tan önce çalışır; böylece doluluk oranı
hatalı değerler NaN yapıldıktan sonra hesaplanır.

Kullanım (proje kök klasöründen):
    python -m src.preprocess
"""
from pathlib import Path

import pandas as pd

from src.data import CITY, POLLUTANTS, ROOT, TARGET, load_city_day

PROCESSED_DIR = ROOT / "data" / "processed"

DROP_COLUMNS = ["City", "AQI", "AQI_Bucket"]  # Şehir tekleşince gereksiz; AQI sızıntı yaratır
MIN_DOLULUK = 0.5  # Bu oranın altında dolu olan kirleticiler çıkarılır
MAX_GAP = 2        # Bu kadar güne kadar süren boşluklar doldurulur
MAX_DEGER = 1000   # µg/m³ üzeri değerler sensör hatası sayılır

FILLED_FLAG = f"{TARGET}_doldurulan"  # Hedefin doldurulan günlerini işaretleyen sütun


def filter_city(df: pd.DataFrame, city: str = CITY) -> pd.DataFrame:
    """Sadece verilen şehrin satırlarını alır.

    - Date sütununu indeks yapar ve tarihe göre sıralar.
    - Döndürür: tek şehre ait, tarih indeksli tablo.
    """
    out = df[df["City"] == city].copy()
    if out.empty:
        raise ValueError(f"'{city}' için hiç satır bulunamadı.")
    out["Date"] = pd.to_datetime(out["Date"])
    out = out.set_index("Date").sort_index()

    tekrar = out.index.duplicated().sum()
    if tekrar:
        print(f"[filter_city] {tekrar} tekrar eden tarih bulundu, ilki tutuldu.")
        out = out[~out.index.duplicated(keep="first")]

    print(f"[filter_city] {city}: {len(out)} satır, "
          f"{out.index.min().date()} - {out.index.max().date()}")
    return out


def complete_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Veride hiç satırı olmayan günleri NaN satır olarak ekler.

    - Başlangıçtan bitişe her günü içeren tam bir takvim oluşturur (asfreq("D")).
    - Kaç gün eklendiğini yazdırır.
    - Neden: Lag özellikleri "bir önceki satır = bir önceki gün" varsayar.
    """
    once = len(df)
    out = df.asfreq("D")
    print(f"[complete_dates] {len(out) - once} eksik gün takvime eklendi.")
    return out


def clean_invalid(df: pd.DataFrame, max_deger: float = MAX_DEGER) -> pd.DataFrame:
    """Fiziksel olarak mümkün olmayan değerleri NaN yapar.

    - Negatif konsantrasyonlar NaN olur.
    - max_deger'in üzerindeki değerler sensör hatası sayılıp NaN olur.
    - Kaç değerin temizlendiğini sütun bazında yazdırır.
    - Değerleri silmez veya doldurmaz; sadece işaretler.
    """
    out = df.copy()
    kolonlar = [c for c in POLLUTANTS if c in out.columns]
    for col in kolonlar:
        gecersiz = (out[col] < 0) | (out[col] > max_deger)
        n = int(gecersiz.sum())
        if n:
            out.loc[gecersiz, col] = pd.NA
            print(f"[clean_invalid] {col}: {n} geçersiz değer NaN yapıldı.")
    return out


def select_columns(df: pd.DataFrame, min_doluluk: float = MIN_DOLULUK) -> pd.DataFrame:
    """Modelde kullanılmayacak sütunları çıkarır.

    - DROP_COLUMNS listesindekileri çıkarır.
    - Doluluk oranı min_doluluk'un altında olan kirleticileri çıkarır.
    - TARGET sütunu doluluğu ne olursa olsun her zaman kalır.
    - Hangi sütunların çıkarıldığını yazdırır.
    """
    out = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])

    doluluk = out.notna().mean()
    dusuk = [c for c in out.columns
             if c != TARGET and doluluk[c] < min_doluluk]
    out = out.drop(columns=dusuk)

    print(f"[select_columns] Çıkarılan sabit sütunlar: "
          f"{[c for c in DROP_COLUMNS if c in df.columns]}")
    if dusuk:
        detay = ", ".join(f"{c} (%{doluluk[c] * 100:.0f})" for c in dusuk)
        print(f"[select_columns] Düşük doluluk nedeniyle çıkarılanlar: {detay}")
    print(f"[select_columns] Kalan sütunlar: {list(out.columns)}")
    return out


def _fill_series(s: pd.Series, max_gap: int) -> tuple[pd.Series, pd.Series]:
    """Tek bir seride sadece max_gap gün veya daha kısa boşlukları doldurur.

    interpolate(limit=...) uzun boşlukların da ilk günlerini doldurduğu için
    her boşluğun uzunluğu ayrıca ölçülür.
    Döndürür: (doldurulmuş seri, doldurulan konumları gösteren maske)
    """
    isna = s.isna()
    grup = (~isna).cumsum()
    bosluk_uzunlugu = isna.groupby(grup).transform("sum")
    doldur = isna & (bosluk_uzunlugu <= max_gap)

    interpolated = s.interpolate(method="time", limit_area="inside")
    doldur = doldur & interpolated.notna()  # serinin başı/sonu doldurulamaz
    return s.where(~doldur, interpolated), doldur


def fill_short_gaps(df: pd.DataFrame, max_gap: int = MAX_GAP) -> pd.DataFrame:
    """Kısa boşlukları doldurur, uzun boşluklara dokunmaz.

    - En fazla max_gap gün süren ardışık boşluklar doldurulur.
    - Daha uzun boşluklar NaN olarak kalır (modelleme aşamasında o günler atılacak).
    - Doldurma yöntemi: zamana göre doğrusal interpolasyon.
    - Her sütunda kaç değerin doldurulduğunu yazdırır.
    - Hedefin doldurulan günleri FILLED_FLAG sütununda işaretlenir. Interpolasyon
      sonraki günün değerini kullandığı için bu günler modelde hedef veya
      değerlendirme günü olarak kullanılmamalıdır.
    """
    out = df.copy()
    sayisal = out.select_dtypes("number").columns
    for col in sayisal:
        out[col], maske = _fill_series(out[col].astype(float), max_gap)
        if maske.any():
            print(f"[fill_short_gaps] {col}: {int(maske.sum())} değer dolduruldu, "
                  f"{int(out[col].isna().sum())} değer boş kaldı.")
        if col == TARGET:
            out[FILLED_FLAG] = maske.astype(int)
    return out


def save_processed(df: pd.DataFrame, city: str = CITY) -> Path:
    """Temiz veriyi data/processed/<sehir>_clean.csv olarak kaydeder.

    - Klasör yoksa oluşturur.
    - Döndürür: kaydedilen dosyanın yolu.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / f"{city.lower()}_clean.csv"
    df.to_csv(path, index_label="Date")
    print(f"[save_processed] Kaydedildi: {path}")
    return path


def run_pipeline(city: str = CITY) -> pd.DataFrame:
    """Tüm ön işleme adımlarını sırayla çalıştırır ve sonucu kaydeder.

    - Döndürür: temizlenmiş tablo.
    """
    df = load_city_day()
    df = filter_city(df, city)
    df = complete_dates(df)
    df = clean_invalid(df)
    df = select_columns(df)
    df = fill_short_gaps(df)
    save_processed(df, city)
    return df


if __name__ == "__main__":
    run_pipeline()