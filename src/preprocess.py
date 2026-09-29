"""Veri ön işleme: city_day.csv'den seçilen şehir için temiz bir günlük seri üretir.

Akış (run_pipeline):
    load_city_day -> filter_city -> complete_dates -> select_columns
                  -> clean_invalid -> fill_short_gaps -> save_processed

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

def filter_city(df: pd.DataFrame, city: str = CITY) -> pd.DataFrame:
    """Sadece verilen şehrin satırlarını alır.

    - Date sütununu indeks yapar ve tarihe göre sıralar.
    - Döndürür: tek şehre ait, tarih indeksli tablo.
    """
    raise NotImplementedError


def complete_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Veride hiç satırı olmayan günleri NaN satır olarak ekler.

    - Başlangıçtan bitişe her günü içeren tam bir takvim oluşturur (asfreq("D")).
    - Kaç gün eklendiğini yazdırır.
    - Neden: Lag özellikleri "bir önceki satır = bir önceki gün" varsayar.
    """
    raise NotImplementedError


def select_columns(df: pd.DataFrame, min_doluluk: float = MIN_DOLULUK) -> pd.DataFrame:
    """Modelde kullanılmayacak sütunları çıkarır.

    - DROP_COLUMNS listesindekileri çıkarır.
    - Doluluk oranı min_doluluk'un altında olan kirleticileri çıkarır.
    - TARGET sütunu doluluğu ne olursa olsun her zaman kalır.
    - Hangi sütunların çıkarıldığını yazdırır.
    """
    raise NotImplementedError


def clean_invalid(df: pd.DataFrame, max_deger: float = MAX_DEGER) -> pd.DataFrame:
    """Fiziksel olarak mümkün olmayan değerleri NaN yapar.

    - Negatif konsantrasyonlar NaN olur.
    - max_deger'in üzerindeki değerler sensör hatası sayılıp NaN olur.
    - Kaç değerin temizlendiğini sütun bazında yazdırır.
    - Değerleri silmez veya doldurmaz; sadece işaretler.
    """
    raise NotImplementedError


def fill_short_gaps(df: pd.DataFrame, max_gap: int = MAX_GAP) -> pd.DataFrame:
    """Kısa boşlukları doldurur, uzun boşluklara dokunmaz.

    - En fazla max_gap gün süren ardışık boşluklar doldurulur.
    - Daha uzun boşluklar NaN olarak kalır (modelleme aşamasında o günler atılacak).
    - Doldurma yöntemi Aşama 3'te tartışılacak (başlangıç: doğrusal interpolasyon).
    - Her sütunda kaç değerin doldurulduğunu yazdırır.
    """
    raise NotImplementedError


def save_processed(df: pd.DataFrame, city: str = CITY) -> Path:
    """Temiz veriyi data/processed/<sehir>_clean.csv olarak kaydeder.

    - Klasör yoksa oluşturur.
    - Döndürür: kaydedilen dosyanın yolu.
    """
    raise NotImplementedError


def run_pipeline(city: str = CITY) -> pd.DataFrame:
    """Tüm ön işleme adımlarını sırayla çalıştırır ve sonucu kaydeder.

    - Döndürür: temizlenmiş tablo.
    """
    raise NotImplementedError


if __name__ == "__main__":
    run_pipeline()