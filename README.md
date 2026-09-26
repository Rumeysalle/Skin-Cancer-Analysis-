# Skin Cancer Analysis & Fairness-Aware Classification (ISLT-HAM10000)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rumeysalle/Skin-Cancer-Analysis-/blob/main/ISLT_HAM10000_v5.ipynb)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c?logo=pytorch&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-HAM10000-blue)
![Model](https://img.shields.io/badge/Model-ResNet50%2BMetadata-brightgreen)
![Fairness](https://img.shields.io/badge/Fairness-ISLT%20%26%20ITA-orange)
![License](https://img.shields.io/badge/License-MIT-green)

Bu proje, **HAM10000** deri kanseri veri seti üzerinde çok modlu (görüntü + klinik demografik veri) derin öğrenme modelleri geliştirir. Amaç iki yönlüdür:

1. Cilt tonu farklılıklarından kaynaklanan model yanlılığını (bias) tespit etmek.
2. **ITA (Individual Typology Angle)** tabanlı **ISLT (ITA-Guided Selective Layer Thawing)** yöntemi ile model adilliğini artırmak.

---

## İçindekiler

- [Öne Çıkan Özellikler](#öne-çıkan-özellikler)
- [Veri Seti ve Sınıf Dağılımı](#veri-seti-ve-sınıf-dağılımı)
- [Metodoloji](#metodoloji)
- [Deney Sonuçları](#deney-sonuçları)
- [Proje Dosya Yapısı](#proje-dosya-yapısı)
- [Kurulum ve Çalıştırma](#kurulum-ve-çalıştırma)
- [Lisans](#lisans)
- [İletişim & Katkı](#i̇letişim--katkı)

---

## Öne Çıkan Özellikler

- **7 lezyon türü sınıflandırması** — Melanom (MEL), Melanositik Nevüs (NV), Bazal Hücreli Karsinom (BCC) ve diğerleri.
- **Çok modlu mimari** — ResNet50 ile çıkarılan görsel temsil, klinik demografik özniteliklerle (yaş, cinsiyet) birleştirilir.
- **Otsu maskeleme ile cilt tipi tespiti** — Lezyon bölgesi Otsu eşikleme ile ayıklanır; sağlıklı deri alanından $L^*a^*b^*$ renk uzayında ITA değeri hesaplanır ve 6 cilt tipine ayrılır (Very Light, Light, Intermediate, Tan, Brown, Dark).
- **Veri sızıntısı önleme** — Veri seti bölünmesi, `lesion_id` baz alınarak hasta düzeyinde yapılır.
- **Grad-CAM disparite analizi ($D_l$)** — Modelin farklı cilt tonlarındaki odaklanma farkını katman bazında ölçer.
- **ISLT ile yanlılık giderme** — XAI metriklerine göre seçilen evrişim katmanlarının dondurulması kaldırılarak (selective thawing) daha adil sınıflandırma sağlanır.

---

## Veri Seti ve Sınıf Dağılımı

**HAM10000** (Human Against Machine with 10000 training images) veri seti toplam **10.015** dermatolojik görüntü içerir:

| Kısaltma | Tanım (Cell Type) | Görüntü Sayısı | Sınıf Kodu |
|---|---|---:|:---:|
| nv | Melanocytic nevi | 6.705 | 4 |
| mel | Melanoma | 1.113 | 6 |
| bkl | Benign keratosis-like lesions | 1.099 | 2 |
| bcc | Basal cell carcinoma | 514 | 1 |
| akiec | Actinic keratoses | 327 | 0 |
| vasc | Vascular lesions | 142 | 5 |
| df | Dermatofibroma | 115 | 3 |

---

## Metodoloji

### 1. ITA (Individual Typology Angle) Hesaplama

Cilt tonu, Otsu tabanlı bir algoritma ile şu adımlarla belirlenir:

1. Görüntü gri seviyeye dönüştürülür, Otsu eşikleme ile lezyon maskesi çıkarılır.
2. Sağlıklı cilt bölgesi belirlenir.
3. Parlaklık ve saturasyon filtrelemesi uygulanır.
4. CIELAB renk uzayındaki $L^*$ ve $b^*$ değerlerinden ITA hesaplanır:

$$\text{ITA} = \arctan\left(\frac{L^* - 50}{b^*}\right) \times \frac{180}{\pi}$$

Hesaplanan ITA değerine göre 6 cilt tonu kategorisi:

| Kategori | ITA Aralığı |
|---|---|
| Very Light | ITA > 55° |
| Light | 41° < ITA ≤ 55° |
| Intermediate | 28° < ITA ≤ 41° |
| Tan | 10° < ITA ≤ 28° |
| Brown | -30° < ITA ≤ 10° |
| Dark | ITA ≤ -30° |

### 2. Mükerrer Kayıt Kontrolü & Veri Bölme

Aynı `lesion_id`'ye sahip birden fazla görüntü bulunabildiğinden, overfitting ve veri sızıntısını önlemek için bölme işlemi hasta bazında yapılır.

### 3. Görüntü Ön İşleme & Normalizasyon

Tüm görüntüler 224×224 boyutuna getirilir ve veri setinden hesaplanan kanal bazlı istatistiklerle normalize edilir:

- **Mean:** `[0.76315, 0.54556, 0.56998]`
- **Std:** `[0.14027, 0.15288, 0.17016]`

### 4. Model Mimarisi (ResNetWithMetadata)

Model üç bileşenden oluşur:

1. **Görsel özellik çıkarıcı** — ImageNet ağırlıklarıyla ilklendirilmiş ResNet50 backbone.
2. **Klinik öznitelik katmanı** — Normalize edilmiş `age` ve `sex_idx` bilgileri.
3. **FC Classifier** — Görsel temsil ile demografik bilgiyi birleştiren tam bağlantılı sınıflandırıcı.

```
Input Image (3x224x224) ──> [ ResNet50 Backbone ] ──> Feature Vector (2048) ┐
                                                                            ├──> [ FC Classifier ] ──> 7 Classes
Input Metadata (Age, Sex) ──────────────────────────> Feature Vector (2)    ┘
```

### 5. ISLT (ITA-Guided Selective Layer Thawing)

Grad-CAM haritalarından hesaplanan disparite skoru ($D_l$) ile, farklı cilt tonlarında karar mekanizmasında sapma gösteren evrişim katmanları tespit edilir. Yalnızca bu katmanlar (örn. `layer4` blokları) eğitime açılarak ince ayar (fine-tuning) yapılır.

---

## Deney Sonuçları

Test veri setinde cilt tipine göre doğruluk (accuracy) karşılaştırması:

| Cilt Tipi | Baseline Accuracy | ISLT Accuracy | Fark |
|---|:---:|:---:|:---:|
| Brown | 100.0% | 100.0% | 0.0% |
| Dark | 100.0% | 100.0% | 0.0% |
| Intermediate | 100.0% | 100.0% | 0.0% |
| Light | 92.86% | 92.86% | 0.0% |
| Very Light | 80.52% | 80.52% | 0.0% |

> **Not:** ISLT yaklaşımı, genel doğruluğu düşürmeden koyu ve ara cilt tiplerindeki yüksek hassasiyeti korur ve katman seviyesindeki Grad-CAM adilliğini belirgin ölçüde iyileştirir.

---

## Proje Dosya Yapısı

```
Skin-Cancer-Analysis-/
├── ISLT_HAM10000_v5.ipynb   # Güncel notebook (ISLT, ITA, Grad-CAM, model eğitimi)
├── ISLT_HAM10000_v4.ipynb   # Önceki notebook versiyonu
├── Ham10000.ipynb           # Temel HAM10000 keşifsel analiz (EDA)
└── README.md                # Proje dokümantasyonu
```

---

## Kurulum ve Çalıştırma

### Gereksinimler

Python 3.10+ ve PyTorch 2.0+ gerekir. Gerekli kütüphaneleri kurun:

```bash
pip install torch torchvision numpy pandas opencv-python scikit-image scikit-learn matplotlib tqdm
```

### Adımlar

1. **Repoyu klonlayın:**
   ```bash
   git clone https://github.com/Rumeysalle/Skin-Cancer-Analysis-.git
   cd Skin-Cancer-Analysis-
   ```

2. **Notebook'u açın:**
   Google Colab veya Jupyter Lab üzerinde `ISLT_HAM10000_v5.ipynb` dosyasını açın.

3. **Veri setini yükleyin:**
   `HAM10000_metadata.csv` ve `HAM10000_images_part_1`, `HAM10000_images_part_2` klasörlerini notebook içindeki ilgili yola (`/content/ham10000`) çıkarın ya da Google Drive entegrasyonunu kullanın.

4. **Eğitim ve değerlendirme:**
   Notebook hücrelerini sırasıyla çalıştırarak ITA hesaplama, baseline model eğitimi, Grad-CAM disparite analizi ve ISLT fine-tuning adımlarını gerçekleştirin.


---

## İletişim & Katkı

Sorularınız, önerileriniz veya katkılarınız için issue açabilir ya da pull request gönderebilirsiniz.
