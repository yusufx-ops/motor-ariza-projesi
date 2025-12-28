"""
Motor Arıza Teşhisi - Test Arayüzü
Streamlit ile MATLAB dosyası yükleyip tahmin yapma
"""
import streamlit as st
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.io import loadmat
from scipy.signal import welch
from scipy.stats import skew, kurtosis
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Sayfa ayarları
st.set_page_config(
    page_title="Motor Arıza Teşhisi",
    page_icon="⚙️",
    layout="wide"
)

# Başlık
st.title("⚙️ Motor Arıza Teşhis Sistemi")
st.markdown("---")

# Model dosyaları (deployment/streamlit/app.py -> ../../models)
MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"

# SABİT İSİMLİ MODEL DOSYALARI
model_path = MODEL_DIR / "paderborn_model.json"
scaler_path = MODEL_DIR / "paderborn_scaler.pkl"
label_encoder_path = MODEL_DIR / "paderborn_label_encoder.pkl"

# Dosyaların varlığını kontrol et
if not model_path.exists():
    st.error(f"❌ Model dosyası bulunamadı: {model_path}")
    st.stop()
if not scaler_path.exists():
    st.error(f"❌ Scaler dosyası bulunamadı: {scaler_path}")
    st.stop()
if not label_encoder_path.exists():
    st.error(f"❌ Label encoder dosyası bulunamadı: {label_encoder_path}")
    st.stop()

st.sidebar.success(f"✅ Model yüklendi: {model_path.name}")

# =====================================================
# FEATURE EXTRACTION FUNCTION
# =====================================================
def extract_vibration_signal(mat_data):
    """
    Paderborn MAT dosyasından vibration sinyalini çıkar
    EĞİTİMDE KULLANILAN AYNI FONKSİYON
    Format: mat[filename][0,0]['Y'] -> structured array
    'vibration_1' field'ını ara ve sinyali döndür
    """
    try:
        # MAT dosyasındaki ana anahtarı bul (__ ile başlamayanlar)
        main_key = None
        for key in mat_data.keys():
            if not key.startswith('__'):
                main_key = key
                break
        
        if main_key is None:
            st.error("❌ MAT dosyasında ana key bulunamadı!")
            return None
        
        st.write(f"📁 Ana key: **{main_key}**")
        
        # Structured data'yı al
        data = mat_data[main_key][0, 0]
        
        # 'Y' field'ındaki tüm sinyalleri kontrol et
        if 'Y' in data.dtype.names:
            signals = data['Y'][0]  # (1, n) array -> n signals
            
            st.write(f"🔍 {len(signals)} adet sinyal bulundu")
            
            # Her sinyali kontrol et
            for idx, signal_data in enumerate(signals):
                # signal_data bir tuple, ilk eleman isim, 3. eleman veri
                if len(signal_data) >= 3:
                    signal_name = signal_data[0]
                    signal_values = signal_data[2]
                    
                    # 'vibration_1' sinyalini ara
                    if isinstance(signal_name, np.ndarray):
                        name_str = str(signal_name[0]) if signal_name.size > 0 else ""
                    else:
                        name_str = str(signal_name)
                    
                    st.write(f"  - Sinyal {idx+1}: {name_str}")
                    
                    if 'vibration_1' in name_str.lower():
                        # Sinyal array'ini düzleştir
                        if isinstance(signal_values, np.ndarray) and signal_values.size > 0:
                            st.success(f"✅ **vibration_1** bulundu! ({signal_values.size} sample)")
                            return signal_values.flatten()
        else:
            st.error("❌ 'Y' field'ı bulunamadı!")
        
        return None
        
    except Exception as e:
        st.error(f"❌ Sinyal çıkarma hatası: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None


def extract_features(signal):
    """Sinyalden 12 feature çıkar (eğitimde kullanılan)"""
    features = {}
    
    # Temel istatistikler
    features['mean'] = np.mean(signal)
    features['peak'] = np.max(np.abs(signal))
    features['min'] = np.min(signal)
    features['max'] = np.max(signal)
    features['range'] = features['max'] - features['min']
    
    # RMS (geçici - crest_factor için)
    rms = np.sqrt(np.mean(signal ** 2))
    
    # Şekil faktörleri
    features['crest_factor'] = features['peak'] / (rms + 1e-10)
    features['kurtosis'] = kurtosis(signal)
    
    # Genlik bazlı
    features['mad'] = np.mean(np.abs(signal - features['mean']))
    features['percentile_25'] = np.percentile(signal, 25)
    features['percentile_75'] = np.percentile(signal, 75)
    features['iqr'] = features['percentile_75'] - features['percentile_25']
    
    # Frekans domain (Spektral)
    freqs, psd = welch(signal, fs=64000, nperseg=min(1024, len(signal)))
    features['spec_centroid'] = np.sum(freqs * psd) / (np.sum(psd) + 1e-10)
    
    return features


# =====================================================
# MODEL YÜKLEME
# =====================================================
@st.cache_resource
def load_model():
    """Model, scaler ve label encoder yükle"""
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    
    return model, scaler, label_encoder


model, scaler, label_encoder = load_model()

st.sidebar.info(f"📊 Model Bilgileri:\n\n- Sınıflar: {', '.join(label_encoder.classes_)}\n- Feature sayısı: 12")

# Sınıf açıklamaları
CLASS_DESCRIPTIONS = {
    'normal': {
        'name': 'NORMAL (ARIZASIZ)',
        'icon': '🟢',
        'color': '#4ECDC4',
        'description': '✅ Rulman sağlıklı durumda, arıza tespit edilmedi.',
        'details': '''
**Durum:** Rulman sağlıklı çalışıyor, herhangi bir arıza belirtisi yok.

**Özellikler:**
- Vibrasyon seviyeleri normal aralıkta
- Spektral merkez (spec_centroid) yüksek
- Crest factor (tepe faktörü) dengeli
- Kurtosis değeri düşük (düzgün dağılım)

**Öneri:** 
- ✅ Normal bakım periyoduna devam edilebilir
- 📅 Rutin kontroller yeterli
- ⚡ Anında müdahale gerekmez
        '''
    },
    'inner': {
        'name': 'İÇ BİLYA ARIZASI (Inner Race Fault)',
        'icon': '🔴',
        'color': '#FF6B6B',
        'description': '⚠️ Rulmanın iç halkasında (inner race) hasar tespit edildi!',
        'details': '''
**Durum:** Rulman iç halkasında aşınma, çatlak veya yüzey hasarı var.

**Özellikler:**
- Mean (ortalama) değer yüksek
- Percentile değerleri artmış
- IQR (çeyrekler arası aralık) genişlemiş
- Vibrasyon amplitüdü artmış

**Nedenleri:**
- ❌ Yetersiz yağlama
- 🔥 Aşırı ısınma
- ⚙️ Yanlış montaj
- 📊 Aşırı yük
- 🕐 Yorulma (fatigue)

**Öneri:**
- 🚨 **ACİL BAKIM GEREKLİ!**
- 🔧 Rulman değiştirilmeli
- ⏱️ Hızla arıza ilerleyebilir
- 💥 Rulman kırılması riski yüksek
        '''
    },
    'outer': {
        'name': 'DIŞ BİLYA ARIZASI (Outer Race Fault)',
        'icon': '🔵',
        'color': '#45B7D1',
        'description': '⚠️ Rulmanın dış halkasında (outer race) hasar tespit edildi.',
        'details': '''
**Durum:** Rulman dış halkasında aşınma, çatlak veya yüzey hasarı var.

**Özellikler:**
- Spec_centroid (spektral merkez) düşük
- Range (değişim aralığı) genişlemiş
- Peak değerler artmış
- Periyodik darbe sinyalleri

**Nedenleri:**
- 🔩 Hatalı montaj (rulman yuvasına oturmamış)
- 💧 Kontaminasyon (kirlenme)
- ⚡ Elektriksel erozyon
- 📐 Hizalama hataları
- 🕒 Yaşlanma

**Öneri:**
- ⚠️ **BAKIM PLANLANMALI**
- 🔍 Düzenli takip gerekli
- 📆 Kısa vadede değişim önerilir
- 🛠️ İç arızaya göre daha yavaş ilerler
- 🔔 Vibrasyon izleme sürdürülmeli
        '''
    }
}

# =====================================================
# DOSYA YÜKLEME
# =====================================================
st.header("📁 MATLAB Dosyası Yükle")

# Toplu dosya yükleme
uploaded_files = st.file_uploader(
    "Paderborn .mat dosyalarını seçin (Birden fazla dosya seçilebilir)",
    type=['mat'],
    help="Dosyalar vibration_1 sinyali içermelidir",
    accept_multiple_files=True
)

if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} dosya yüklendi")
    
    # Toplu analiz için sonuç listesi
    results = []
    
    # Progress bar için placeholder
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Her dosya için işlem
    for file_idx, uploaded_file in enumerate(uploaded_files):
        # Progress güncelle
        progress = (file_idx + 1) / len(uploaded_files)
        progress_bar.progress(progress)
        status_text.text(f"İşleniyor: {file_idx + 1}/{len(uploaded_files)} - {uploaded_file.name}")
        
        st.markdown("---")
        st.subheader(f"📄 Dosya {file_idx + 1}/{len(uploaded_files)}: {uploaded_file.name}")
        
        # MATLAB dosyasını oku
        try:
            mat_data = loadmat(uploaded_file)
            
            # Sinyal çıkar
            signal = extract_vibration_signal(mat_data)
            
            if signal is None or len(signal) < 100:
                st.error(f"❌ {uploaded_file.name}: Geçerli vibration sinyali bulunamadı!")
                continue
            
            # =====================================================
            # SİNYAL ANALİZİ
            # =====================================================
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📊 Sinyal Uzunluğu", f"{len(signal):,} sample")
            with col2:
                st.metric("⏱️ Süre", f"{len(signal)/64000:.2f} saniye")
            with col3:
                st.metric("📈 Örnekleme Frekansı", "64 kHz")
            
            # Sinyal görselleştirme (sadece ilk dosya için)
            if file_idx == 0 or len(uploaded_files) == 1:
                st.write("**📈 Vibrasyon Sinyali**")
                
                fig, ax = plt.subplots(figsize=(12, 3))
                time = np.arange(len(signal)) / 64000
                ax.plot(time[:5000], signal[:5000], linewidth=0.8, color='steelblue')
                ax.set_xlabel('Zaman (saniye)', fontsize=10)
                ax.set_ylabel('Genlik', fontsize=10)
                ax.set_title('Vibrasyon Sinyali (İlk 5000 sample)', fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()
            
            # =====================================================
            # FEATURE ÇIKARIMI
            # =====================================================
            with st.spinner("Feature'lar hesaplanıyor..."):
                features = extract_features(signal)
            
            # Feature DataFrame
            feature_df = pd.DataFrame([features])
            
            # Feature sırasını doğru şekilde düzenle
            feature_order = ['mean', 'peak', 'min', 'max', 'range', 'crest_factor', 
                            'kurtosis', 'mad', 'percentile_25', 'percentile_75', 
                            'iqr', 'spec_centroid']
            
            X = feature_df[feature_order].values
            
            # Standardization
            X_scaled = scaler.transform(X)
            
            # Tahmin
            prediction = model.predict(X_scaled)[0]
            prediction_proba = model.predict_proba(X_scaled)[0]
            
            predicted_label = label_encoder.inverse_transform([prediction])[0]
            
            # Label encoder sırasına göre index bul
            class_indices = {label: idx for idx, label in enumerate(label_encoder.classes_)}
            
            # Sonucu kaydet
            results.append({
                'Dosya': uploaded_file.name,
                'Tahmin': predicted_label,
                'Güven (%)': prediction_proba[prediction] * 100,
                'Inner (%)': prediction_proba[class_indices['inner']] * 100,
                'Normal (%)': prediction_proba[class_indices['normal']] * 100,
                'Outer (%)': prediction_proba[class_indices['outer']] * 100
            })
            
            # Sonuçları göster
            class_info = CLASS_DESCRIPTIONS[predicted_label]
            
            st.markdown(f"""
            <div style="background-color: {class_info['color']}; padding: 20px; border-radius: 10px; margin: 15px 0;">
                <h2 style="color: white; margin: 0;">{class_info['icon']} {class_info['name']}</h2>
                <h3 style="color: white; margin-top: 10px;">Güven: {prediction_proba[prediction]*100:.2f}%</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Açıklama
            st.info(f"**{class_info['description']}**\n\n{class_info['details']}")
            
            # Olasılık dağılımı (sadece tek dosya veya son dosya için)
            if len(uploaded_files) == 1 or file_idx == len(uploaded_files) - 1:
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.write("**📊 Sınıf Olasılıkları:**")
                    proba_df = pd.DataFrame({
                        'Sınıf': label_encoder.classes_,
                        'Olasılık (%)': prediction_proba * 100
                    }).sort_values('Olasılık (%)', ascending=False)
                    
                    st.dataframe(proba_df.style.format({'Olasılık (%)': '{:.2f}%'}), 
                                use_container_width=True, hide_index=True)
                
                with col2:
                    st.write("**📈 Olasılık Grafiği:**")
                    fig2, ax2 = plt.subplots(figsize=(6, 4))
                    colors_map = {'inner': '#FF6B6B', 'normal': '#4ECDC4', 'outer': '#45B7D1'}
                    colors_list = [colors_map[label] for label in label_encoder.classes_]
                    
                    bars = ax2.bar(label_encoder.classes_, prediction_proba * 100, 
                                  color=colors_list, alpha=0.8, edgecolor='black', linewidth=2)
                    
                    # Bar üstüne değerleri yaz
                    for bar, val in zip(bars, prediction_proba * 100):
                        height = bar.get_height()
                        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
                    
                    ax2.set_ylabel('Olasılık (%)', fontsize=10)
                    ax2.set_title('Tahmin Olasılıkları', fontsize=12, fontweight='bold')
                    ax2.set_ylim([0, 105])
                    ax2.grid(True, alpha=0.3, axis='y')
                    plt.tight_layout()
                    st.pyplot(fig2)
                    plt.close()
            
        except Exception as e:
            st.error(f"❌ {uploaded_file.name}: Hata oluştu - {e}")
            continue
    
    # Progress bar'ı temizle
    progress_bar.empty()
    status_text.empty()
    
    # Toplu sonuç özeti
    if len(results) > 1:
        st.markdown("---")
        st.header("📊 Toplu Analiz Özeti")
        
        results_df = pd.DataFrame(results)
        
        # Özet istatistikler
        col1, col2, col3 = st.columns(3)
        
        with col1:
            normal_count = (results_df['Tahmin'] == 'normal').sum()
            st.metric("🟢 Normal", f"{normal_count} dosya", 
                     f"{normal_count/len(results)*100:.1f}%")
        
        with col2:
            inner_count = (results_df['Tahmin'] == 'inner').sum()
            st.metric("🔴 Inner Arıza", f"{inner_count} dosya", 
                     f"{inner_count/len(results)*100:.1f}%")
        
        with col3:
            outer_count = (results_df['Tahmin'] == 'outer').sum()
            st.metric("🔵 Outer Arıza", f"{outer_count} dosya", 
                     f"{outer_count/len(results)*100:.1f}%")
        
        # Detaylı tablo
        st.subheader("📋 Detaylı Sonuçlar")
        st.dataframe(
            results_df.style.format({
                'Güven (%)': '{:.2f}%',
                'Inner (%)': '{:.2f}%',
                'Normal (%)': '{:.2f}%',
                'Outer (%)': '{:.2f}%'
            }).background_gradient(subset=['Güven (%)'], cmap='RdYlGn'),
            use_container_width=True,
            hide_index=True
        )
        
        # Pie Chart - Sınıf Dağılımı
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🥧 Sınıf Dağılımı")
            fig_pie, ax_pie = plt.subplots(figsize=(8, 6))
            
            counts = [normal_count, inner_count, outer_count]
            labels = ['🟢 Normal', '🔴 Inner Arıza', '🔵 Outer Arıza']
            colors = ['#4ECDC4', '#FF6B6B', '#45B7D1']
            explode = (0.05, 0.05, 0.05)
            
            wedges, texts, autotexts = ax_pie.pie(
                counts, 
                labels=labels, 
                colors=colors,
                autopct='%1.1f%%',
                startangle=90,
                explode=explode,
                shadow=True,
                textprops={'fontsize': 12, 'weight': 'bold'}
            )
            
            # Yüzde metinlerini beyaz yap
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(14)
            
            ax_pie.set_title(f'Toplam {len(results)} Dosya Analizi', 
                           fontsize=14, fontweight='bold', pad=20)
            
            plt.tight_layout()
            st.pyplot(fig_pie)
            plt.close()
        
        with col2:
            st.subheader("📊 İstatistikler")
            
            # Ortalama güven seviyesi
            avg_confidence = results_df['Güven (%)'].mean()
            st.metric("📈 Ortalama Güven", f"{avg_confidence:.2f}%")
            
            # En düşük güven
            min_confidence = results_df['Güven (%)'].min()
            min_file = results_df.loc[results_df['Güven (%)'].idxmin(), 'Dosya']
            st.metric("⚠️ En Düşük Güven", f"{min_confidence:.2f}%", 
                     f"{min_file[:30]}...")
            
            # En yüksek güven
            max_confidence = results_df['Güven (%)'].max()
            max_file = results_df.loc[results_df['Güven (%)'].idxmax(), 'Dosya']
            st.metric("✅ En Yüksek Güven", f"{max_confidence:.2f}%",
                     f"{max_file[:30]}...")
            
            # Arıza oranı
            fault_ratio = ((inner_count + outer_count) / len(results)) * 100
            st.metric("🔧 Toplam Arıza Oranı", f"{fault_ratio:.1f}%",
                     f"{inner_count + outer_count}/{len(results)} dosya")
        
        # CSV İndirme Butonu
        st.markdown("---")
        st.subheader("💾 Sonuçları İndir")
        
        # CSV formatına dönüştür
        csv = results_df.to_csv(index=False, encoding='utf-8-sig')
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.download_button(
                label="📥 CSV İndir",
                data=csv,
                file_name=f"motor_ariza_analiz_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                help="Sonuçları CSV formatında indir"
            )
        
        with col2:
            # Özet rapor oluştur
            summary_text = f"""MOTOR ARIZA TEŞHİS RAPORU
{'='*50}
Tarih: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Toplam Dosya: {len(results)}

SINIF DAĞILIMI:
- Normal (Arızasız): {normal_count} dosya ({normal_count/len(results)*100:.1f}%)
- Inner Arıza: {inner_count} dosya ({inner_count/len(results)*100:.1f}%)
- Outer Arıza: {outer_count} dosya ({outer_count/len(results)*100:.1f}%)

İSTATİSTİKLER:
- Ortalama Güven: {avg_confidence:.2f}%
- En Düşük Güven: {min_confidence:.2f}% ({min_file})
- En Yüksek Güven: {max_confidence:.2f}% ({max_file})
- Toplam Arıza Oranı: {fault_ratio:.1f}%

{'='*50}

DETAYLI SONUÇLAR:
"""
            for idx, row in results_df.iterrows():
                summary_text += f"\n{idx+1}. {row['Dosya']}\n"
                summary_text += f"   Tahmin: {row['Tahmin'].upper()} (Güven: {row['Güven (%)']:.2f}%)\n"
                summary_text += f"   Inner: {row['Inner (%)']:.2f}% | Normal: {row['Normal (%)']:.2f}% | Outer: {row['Outer (%)']:.2f}%\n"
            
            st.download_button(
                label="📄 TXT Rapor İndir",
                data=summary_text,
                file_name=f"motor_ariza_rapor_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                help="Özet raporu TXT formatında indir"
            )

else:
    st.info("👆 Lütfen bir MATLAB dosyası yükleyin")
    
    # Örnek dosya yapısı göster
    with st.expander("ℹ️ Dosya Formatı Bilgisi"):
        st.markdown("""
        **Beklenen MATLAB Dosya Yapısı:**
        
        - Dosya `.mat` formatında olmalı
        - İçinde `vibration_1` adlı bir sinyal bulunmalı
        - Sinyal numerik array formatında olmalı
        - Örnekleme frekansı: 64 kHz (önerilir)
        
        **Örnek Dosya Yolu:**
        ```
        data/paderborn_raw/inner/KI01/N15_M01_F10_KI01_1.mat
        ```
        """)
