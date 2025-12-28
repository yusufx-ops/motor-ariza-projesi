# Motor Arıza Teşhis Sistemi

XGBoost tabanlı rulman arıza tespit sistemi. Paderborn dataset kullanılarak geliştirilmiştir.

## 🎯 Özellikler

- **İç Bilya Arızası (Inner Race)** tespiti
- **Dış Bilya Arızası (Outer Race)** tespiti  
- **Normal (Arızasız)** rulman tespiti
- Toplu dosya analizi
- CSV/TXT rapor indirme
- %96+ doğruluk oranı

## 🚀 Canlı Demo

[Streamlit Cloud'da dene](#) _(link eklenecek)_

## 📊 Kullanım

1. MATLAB (.mat) dosyanızı yükleyin
2. Sistem otomatik analiz yapar
3. Tahmin ve güven yüzdesini görün
4. Toplu analiz için birden fazla dosya yükleyin

## 🛠️ Yerel Kurulum

```bash
pip install -r requirements.txt
streamlit run deployment/streamlit/app.py
```

## 📈 Model Performansı

- Accuracy: 96.46%
- Overfitting: Düşük (0.94% gap)
- Features: 12 adet (mean, peak, spec_centroid, vb.)

## 📁 Proje Yapısı

- `deployment/streamlit/` - Web arayüzü
- `models/` - Eğitilmiş model dosyaları
- `analysis/` - Görselleştirme scriptleri

## 👨‍💻 Geliştirici

Yusuf - Motor Arıza Teşhis Projesi
