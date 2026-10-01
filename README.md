# Hava Kalitesi Tahmini: Delhi PM2.5

Delhi için **yarının günlük PM2.5 konsantrasyonunu** geçmiş kirlilik verilerinden tahmin eden bir makine öğrenmesi projesi. LightGBM, Ridge regresyon ve SARIMAX modelleri, basit bir "yarın = bugün" kuralıyla karşılaştırılıyor.

## Veri

[Air Quality Data in India (2015–2020)](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india) veri setindeki `city_day.csv` dosyası kullanılıyor. Dosya Hindistan'daki şehirlerin günlük kirletici ölçümlerini (PM2.5, PM10, NO2, CO, O3 vb.) ve AQI değerlerini içeriyor. Bu projede sadece **Delhi** verisi ve hedef olarak **PM2.5** kullanılıyor; yaklaşık 2000 günlük bir seri (Ocak 2015 – Temmuz 2020).

Veri dosyası repoda bulunmuyor. Kaggle'dan indirip `data/raw/city_day.csv` olarak kaydedin.

## Kurulum

```bash
python -m venv .venv
source .venv/Scripts/activate      # Windows Git Bash
# .venv\Scripts\Activate.ps1       # Windows PowerShell
pip install pandas numpy scikit-learn lightgbm statsmodels matplotlib
```

## Kullanım

Tüm komutlar proje kök klasöründen ve `python -m` ile çalıştırılmalıdır; scriptler `src` paketinden içe aktarım yapıyor.

```bash
python -m src.eda                  # keşifsel veri analizi
python -m src.preprocess           # temiz veriyi data/processed/ altına üretir
python -m src.train_lightgbm       # LightGBM'i eğitir, 2019 doğrulamada ölçer
python -m src.train_ridge          # Ridge ve LightGBM'i yan yana karşılaştırır
python -m src.train_sarimax        # SARIMAX'ı eğitir, 2019 doğrulamada ölçer
```

Eğitim scriptleri `--final` parametresiyle train ve doğrulama verisinde yeniden eğitilip 2020 test setinde değerlendirilir. Test seti yalnızca tüm denemeler bittikten sonra bir kez kullanılmalıdır.

## Proje yapısı

```
src/
├── data.py             # ortak sabitler (şehir, hedef, kirleticiler) ve veri okuma
├── eda.py              # keşifsel veri analizi
├── preprocess.py       # ön işleme akışı
├── train_lightgbm.py   # LightGBM eğitimi ve değerlendirmesi
├── train_ridge.py      # Ridge regresyon ve model karşılaştırması
└── train_sarimax.py    # SARIMAX eğitimi ve değerlendirmesi
```

## Yöntem

### Ön işleme

`src/preprocess.py` şu adımları sırayla uygular:

1. Delhi satırlarını seçer ve tarihe göre sıralar.
2. Veride hiç bulunmayan günleri boş satır olarak takvime ekler. Böylece "bir önceki satır = bir önceki gün" varsayımı her zaman geçerli olur.
3. Negatif ve 1000 µg/m³ üzerindeki fiziksel olarak mümkün olmayan değerleri boş olarak işaretler.
4. AQI ve AQI_Bucket sütunlarını çıkarır (AQI, PM2.5'ten hesaplandığı için sızıntı yaratır) ve %50'den az dolu kirleticileri atar.
5. En fazla 2 gün süren boşlukları zamana göre doğrusal interpolasyonla doldurur; daha uzun boşluklara dokunmaz. PM2.5'in doldurulan günleri ayrı bir sütunda işaretlenir.

### Özellikler (LightGBM ve Ridge)

Her satır "bugün", hedef ise "yarının PM2.5 değeri". Tüm özellikler yalnızca bugün ve öncesindeki bilgiyi kullanır:

- PM2.5'in son 1, 2, 3, 7 ve 14 gündeki değerleri
- 3, 7 ve 14 günlük hareketli ortalamalar, 7 günlük standart sapma ve günlük değişim
- Diğer kirleticilerin bugünkü değerleri
- Hedef günün ayı, haftanın günü ve yıl içindeki konumu (sinüs/kosinüs)

### Veri sızıntısına karşı önlemler

- Veri **zamana göre** bölünür: 2015–2018 eğitim, 2019 doğrulama, 2020 (Ocak–Temmuz) test. Rastgele bölme, birbirine çok benzeyen ardışık günleri eğitim ve teste dağıtacağı için iyimser skorlar üretirdi.
- İnterpolasyonla doldurulan bir gün, sonraki günün gerçek değerini içerir. LightGBM ve Ridge'de hedef günü veya hedefe çok yakın PM2.5 değeri doldurulmuş satırlar eğitimden ve ölçümden çıkarılır. SARIMAX eksik günleri kendisi yönetebildiği için doldurulan günler bu modelde tekrar boş bırakılır.
- Ridge için eksik değer doldurma ve ölçekleme parametreleri yalnızca eğitim verisinden öğrenilir.
- Tüm modeller aynı günler üzerinde değerlendirilir; SARIMAX'ın her günkü tahmini yalnızca bir önceki güne kadarki gerçek değerleri kullanır.

### Modeller

- **Persistence (baseline):** Yarının değeri bugünkü değere eşit kabul edilir. Bir modelin işe yaradığını söyleyebilmek için bu kuralı geçmesi gerekir.
- **Ridge regresyon:** Doğrusal model. Ceza parametresi `alpha` doğrulama setinde seçilir.
- **LightGBM:** Gradient boosting tabanlı ağaç modeli. Veri küçük olduğu için sade ağaçlar (`num_leaves=15`) ve early stopping kullanılır.
- **SARIMAX:** Klasik istatistiksel zaman serisi modeli. Diğer modellerden farklı olarak yalnızca PM2.5'in kendi geçmişini kullanır; başka kirleticilerden bilgi almaz.
  - Hedef `log(1 + PM2.5)` dönüşümüyle modellenir, çünkü dağılım sağa çarpıktır.
  - `SARIMAX(2,0,1)x(1,0,1,7)`: son iki günün değeri ve bir önceki günün tahmin hatası, ayrıca haftalık mevsimsellik.
  - Yıllık mevsimsellik, 365 günlük bir mevsimsel periyot pratikte modellenemediği için iki sinüs/kosinüs çiftinden oluşan Fourier terimleriyle dış değişken olarak eklenir. Bu terimler içinde en büyük katsayı pozitif `cos1` (0.68); yani model kış aylarında kirliliğin yükseldiğini öğrenmiş.
  - Tahminlerle birlikte %95 güven aralığı da üretir.

## Sonuçlar

### Doğrulama seti (2019, 365 gün)

| Model | MAE | RMSE | R² |
| --- | --- | --- | --- |
| Persistence (yarın = bugün) | 27.95 | 46.38 | 0.71 |
| Ridge | 27.33 | 43.30 | 0.74 |
| SARIMAX | **26.14** | 41.99 | 0.76 |
| LightGBM | 26.53 | **40.20** | **0.78** |

Hatalar µg/m³ cinsindendir. Her sütundaki en iyi değer kalın olarak gösterilmiştir.

### Aylara göre MAE (2019)

| Ay | Persistence | Ridge | SARIMAX | LightGBM |
| --- | --- | --- | --- | --- |
| Ocak | 56.5 | **48.7** | 52.4 | 50.4 |
| Şubat | 31.0 | 30.7 | **26.6** | 33.1 |
| Mart | 20.3 | 18.8 | **18.1** | 18.3 |
| Nisan | 16.6 | 16.1 | **15.3** | 17.3 |
| Mayıs | 23.2 | 23.4 | 21.6 | **20.8** |
| Haziran | 18.8 | **17.6** | 17.9 | 18.2 |
| Temmuz | 10.8 | 10.7 | **10.3** | 11.8 |
| Ağustos | 8.3 | 8.2 | **7.6** | 9.3 |
| Eylül | **12.0** | 16.1 | 12.7 | 12.7 |
| Ekim | **22.6** | 24.1 | 23.2 | 23.1 |
| Kasım | 68.2 | 68.4 | 66.5 | **63.6** |
| Aralık | 47.4 | 45.7 | 41.8 | **40.6** |

Her ay için en düşük hata kalın olarak gösterilmiştir.

### Bulgular

- **Tek bir "en iyi" model yok; hangi ölçüye baktığınıza bağlı.** SARIMAX en düşük MAE'yi (26.14), LightGBM en düşük RMSE'yi (40.20) ve en yüksek R²'yi (0.78) veriyor. Ridge her üç ölçüde de ikisinin gerisinde kalıyor. Tüm modeller persistence'ı geçiyor, ama iyileşme sınırlı: MAE'de %2–6.5, RMSE'de %7–13.
- **SARIMAX sıradan günlerde, LightGBM zirvelerde daha iyi.** SARIMAX, Şubat–Nisan ve Temmuz–Ağustos gibi sakin ve orta kirlilik dönemlerinde en düşük hatayı veriyor. LightGBM ise Kasım ve Aralık'taki yüksek kirlilik dönemlerinde öne çıkıyor. MAE her hatayı eşit sayarken RMSE büyük hataları ağır cezalandırdığı için, iki ölçünün farklı modelleri öne çıkarması bu tabloyla tutarlı.
- **SARIMAX'ın MAE üstünlüğünün bir kısmı log dönüşümünden geliyor.** Log ölçeğindeki tahmin geri dönüştürüldüğünde ortalamayı değil ortancayı verir. Ortanca, mutlak hatayı en aza indiren değer olduğu için bu durum MAE'yi iyileştirirken zirvelerde tahminin düşük kalmasına ve RMSE'nin artmasına yol açıyor.
- **Ek kirleticiler pek bilgi eklemiyor.** SARIMAX yalnızca PM2.5'in kendi geçmişini ve takvimi kullandığı halde, diğer kirleticilere de erişen Ridge'i her ölçüde geçiyor. Bu, tahmin için gereken bilginin büyük kısmının PM2.5'in kendi geçmişinde olduğunu gösteriyor.
- **Hataların büyük kısmı kış aylarından geliyor.** Kasım'daki hata Ağustos'takinin 7–9 katı. Bu dönemde anız yakma, Diwali ve durgun hava koşulları ani sıçramalar yaratıyor; hiçbir model bu sıçramaları önceden yakalayamıyor.
- **Modeller büyük ölçüde bugünkü değere dayanıyor.** LightGBM'de en önemli özellik bugünkü PM2.5 (toplam katkının yaklaşık üçte ikisi); bunu kısa dönem ortalamalar ve mevsimsellik izliyor. Modeller uç değerleri ortalamaya doğru çekiyor ve zirveleri genellikle bir gün geriden takip ediyor.

## Sınırlılıklar ve sonraki adımlar

- **Meteorolojik veri yok.** Yarının kirliliğini büyük ölçüde rüzgar, sıcaklık, nem ve yağış belirliyor. Bu bilgiler olmadan modeller ani değişimleri önceden göremiyor. Geçmiş hava durumu verisinin (örneğin Open-Meteo Historical Weather API) eklenmesi en büyük iyileşmeyi sağlayabilir. Bu durumda eğitimde gerçekleşen hava durumunun kullanılmasının, gerçek kullanımdaki hava tahmini hatasını gizleyeceği göz önünde bulundurulmalı.
- **Test dönemi temsil gücü sınırlı.** 2020 test seti yalnızca Ocak–Temmuz aylarını kapsıyor; en zor aylar olan Ekim–Aralık bulunmuyor ve Mart sonundan itibaren pandemi kapanması etkisi var. Test skorları bu nedenle doğrulama skorlarından iyi çıkabilir.
- **Ridge katsayıları doğrudan yorumlanamaz.** Hareketli ortalama ve günlük değişim özellikleri lag değerlerinden türetildiği için katsayılar arasında çoklu doğrusal bağlantı var.
- **SARIMAX tahminleri hafif aşağı yönlü.** Log ölçeğinden geri dönüştürülen tahmin ortancayı verdiği için, sağa çarpık bir dağılımda tahminler sistematik olarak biraz düşük kalıyor.
- **SARIMAX'ın güven aralıkları olması gerekenden dar.** %95 güven aralığı gerçek değerlerin yalnızca %89'unu kapsıyor (ortanca genişlik 97 µg/m³). Model, özellikle ani zirvelerdeki belirsizliği olduğundan az tahmin ediyor.
- **Haftalık mevsimsel bileşen işlevsiz görünüyor.** Mevsimsel AR (0.998) ve MA (−0.994) katsayıları birbirini neredeyse tamamen götürüyor. Bu, veride belirgin bir haftalık desen olmadığını düşündürüyor; mevsimsel kısım olmadan daha sade bir `SARIMAX(2,0,1)` modeli denenebilir.

## Katkıda bulunanlar

- Erva Nur Bostancı ([@Ervanurb](https://github.com/Ervanurb))
- Dide Ozguven ([@dideozguven](https://github.com/dideozguven))