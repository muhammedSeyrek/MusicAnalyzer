import streamlit as st
import os
import json
import time
import threading
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from collections import Counter

# Sayfa Başlığı ve Açıklama
st.set_page_config(page_title="Müzik Analiz Uygulaması", layout="wide")
st.title("Müzik Analiz Uygulaması")
st.write("Bir müzik dosyası yükleyin ve analiz sonuçlarını görüntüleyin.")

# Klasör yapısını oluştur
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Analiz fonksiyonları
def detect_tonality(freqs):
    """
    Detect tonality by comparing frequency ratios to known musical systems
    """
    # Define Western major and minor scales
    western_ratios = {
        'C Major': [1.122, 1.260, 1.335, 1.498, 1.682, 1.888, 2.0],
        'G Major': [1.125, 1.265, 1.333, 1.500, 1.687, 1.895, 2.0],
        'D Major': [1.120, 1.259, 1.336, 1.496, 1.680, 1.890, 2.0],
        'A Major': [1.123, 1.262, 1.334, 1.499, 1.685, 1.886, 2.0],
        'E Major': [1.121, 1.258, 1.337, 1.497, 1.683, 1.885, 2.0],
        'A Minor': [1.122, 1.189, 1.335, 1.498, 1.587, 1.782, 2.0],
        'E Minor': [1.120, 1.187, 1.337, 1.497, 1.585, 1.780, 2.0],
        'B Minor': [1.123, 1.190, 1.334, 1.499, 1.589, 1.784, 2.0]
    }
    
    # Define Eastern makams (Turkish music)
    eastern_ratios = {
        'Hicaz': [1.08, 1.13, 1.25, 1.33, 1.5, 1.67, 1.8],
        'Rast': [1.12, 1.25, 1.33, 1.5, 1.67, 1.87, 2.0],
        'Nihavend': [1.11, 1.18, 1.33, 1.5, 1.59, 1.78, 2.0],
        'Hüseyni': [1.13, 1.25, 1.35, 1.5, 1.68, 1.8, 2.0],
        'Segah': [1.14, 1.2, 1.32, 1.5, 1.66, 1.78, 2.0],
        'Uşşak': [1.13, 1.25, 1.35, 1.5, 1.68, 1.8, 2.0],
        'Saba': [1.11, 1.19, 1.31, 1.42, 1.59, 1.75, 2.0]
    }
    
    # Filter out extreme values and zero frequencies
    freqs = [f for f in freqs if 20 < f < 20000]
    
    if len(freqs) < 8:
        return {
            'western_tonality': 'Unknown',
            'eastern_makam': 'Unknown',
            'is_western': True,  # Default to Western when uncertain
            'western_confidence': 0.5,
            'eastern_confidence': 0.5
        }
    
    # Calculate frequency ratios between consecutive notes
    ratios = []
    for i in range(len(freqs)-1):
        r = freqs[i+1] / freqs[i]
        # Filter out extreme ratios
        if 1.0 < r < 2.1:
            ratios.append(r)
    
    # Find the closest musical system
    def closest_system(ratios, systems):
        errors = {}
        for name, system_ratios in systems.items():
            # Calculate how well the observed ratios match this system
            error = sum(min(abs(r - sr) for sr in system_ratios) for r in ratios)
            errors[name] = error / max(1, len(ratios))
        
        # Return the system with minimum error
        if errors:
            return min(errors.items(), key=lambda x: x[1])
        return ('Unknown', float('inf'))
    
    # Find the best matching western and eastern systems
    western_best = closest_system(ratios, western_ratios)
    eastern_best = closest_system(ratios, eastern_ratios)
    
    # Determine if the music is more western or eastern
    is_western = western_best[1] < eastern_best[1]
    
    return {
        'western_tonality': western_best[0],
        'eastern_makam': eastern_best[0],
        'is_western': is_western,
        'western_confidence': 1.0 / (1.0 + western_best[1]),
        'eastern_confidence': 1.0 / (1.0 + eastern_best[1])
    }

def analyze_music(filepath, progress_callback=None):
    """
    Main function to analyze the music file
    """
    # If progress callback exists, start reporting
    if progress_callback:
        progress_callback(10)  # Initial progress
    
    # Load the audio file
    try:
        # Dosya boyutunu kontrol et ve gerekirse örnekleme oranını düşür
        file_size = os.path.getsize(filepath)
        sr_target = None  # Varsayılan örnekleme oranı
        
        # Büyük dosyalar için örnekleme oranını düşür
        if file_size > 10 * 1024 * 1024:  # 10MB'dan büyük dosyalar
            sr_target = 22050  # Düşük örnekleme oranı
        
        # Dosyayı kısa bir segment olarak yükle (ilk 60 saniye)
        y, sr = librosa.load(filepath, sr=sr_target, duration=60)
        
        if progress_callback:
            progress_callback(30)  # Audio loaded
        
        # Basic audio properties
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Extract pitches - düşük çözünürlükte analiz
        n_fft = 2048  # Daha küçük FFT penceresi
        hop_length = 1024  # Daha büyük hop size
        
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        
        if progress_callback:
            progress_callback(50)  # Pitch analysis done
        
        # Daha az sayıda frekans al (performans için)
        freqs = []
        max_frames = min(pitches.shape[1], 100)  # En fazla 100 çerçeve
        step = max(1, pitches.shape[1] // max_frames)
        
        for t in range(0, pitches.shape[1], step):
            # Get the frequency with the highest magnitude at this time frame
            if magnitudes[:, t].max() > 0:
                index = magnitudes[:, t].argmax()
                freq = pitches[index, t]
                if freq > 0:  # Only include non-zero frequencies
                    freqs.append(freq)
        
        # Detect tonality
        tonality = detect_tonality(freqs)
        
        if progress_callback:
            progress_callback(70)  # Tonality analysis done
        
        # Analyze rhythm - basitleştirilmiş analiz
        # Sadece onset_env hesapla, daha az veri işle
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        
        rhythm_info = {
            "tempo": tempo,
            "beat_regularity": 0.8,  # Sabit bir değer kullan
            "rhythm_pattern": "4/4"  # Varsayılan değer
        }
        
        if progress_callback:
            progress_callback(80)  # Rhythm analysis done
        
        # Analyze timbre - basitleştirilmiş analiz
        # Daha az MFCC özelliği hesapla
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=8)
        mfcc_mean = np.mean(mfcc, axis=1)
        
        # Spectral özellikleri hesapla
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        brightness = np.mean(spectral_centroid) / (sr/2)
        
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        richness = np.mean(contrast)
        
        # Basit enstrüman tahmini
        if brightness > 0.4 and richness > 5:
            instrument_family = "Brass"
        elif brightness > 0.3 and richness > 3:
            instrument_family = "String"
        elif brightness < 0.2 and richness > 4:
            instrument_family = "Percussion"
        elif brightness < 0.3 and richness < 3:
            instrument_family = "Woodwind"
        else:
            instrument_family = "Mixed"
            
        timbre_info = {
            "brightness": float(brightness),
            "richness": float(richness),
            "instrument_family": instrument_family,
            "mfcc_features": mfcc_mean.tolist()
        }
        
        if progress_callback:
            progress_callback(90)  # Timbre analysis done
        
        # Combine all results
        result = {
            'duration': float(duration),
            'sample_rate': int(sr),
            'tempo': float(rhythm_info['tempo']),
            'beat_regularity': float(rhythm_info['beat_regularity']),
            'rhythm_pattern': str(rhythm_info['rhythm_pattern']),
            'tonality': {
                'western_tonality': str(tonality['western_tonality']),
                'eastern_makam': str(tonality['eastern_makam']),
                'is_western': bool(tonality['is_western']),
                'western_confidence': float(tonality.get('western_confidence', 0.5)),
                'eastern_confidence': float(tonality.get('eastern_confidence', 0.5))
            },
            'timbre': {
                'brightness': float(timbre_info['brightness']),
                'richness': float(richness),
                'instrument_family': str(timbre_info['instrument_family']),
                'mfcc_features': [float(x) for x in timbre_info['mfcc_features']]
            },
            'frequencies': [float(f) for f in freqs[:50]],  # Sadece ilk 50 frekans
            'system': 'Batı' if bool(tonality['is_western']) else 'Doğu',
            'audio_data': {
                'y': y.tolist()[:10000],  # Just a small sample for visualization
                'sr': sr
            }
        }
        
        if progress_callback:
            progress_callback(100)  # Analysis complete
        
        return result
    
    except Exception as e:
        print(f"Error analyzing music: {e}")
        import traceback
        traceback.print_exc()
        
        # Return a basic error result
        return {
            'error': str(e),
            'system': 'Unknown',
            'tonality': {
                'western_tonality': 'Unknown',
                'eastern_makam': 'Unknown',
                'is_western': True
            },
            'frequencies': [],
            'tempo': 120.0  # Default tempo
        }

# JSON Serileştirme için yardımcı fonksiyon
def custom_serializer(obj):
    if isinstance(obj, (bool, int, float, str)):
        return obj
    # NumPy tiplerine özel dönüşüm
    if hasattr(obj, 'item'):
        return obj.item()
    # Diğer iterable nesneler için liste dönüşümü yap
    try:
        return list(obj)
    except:
        return str(obj)

# Görselleştirme fonksiyonları
def create_waveform(y, sr):
    """
    Create a waveform visualization of the audio
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    # prop_cycler hatası giderildi - renk manuel olarak belirleniyor
    librosa.display.waveshow(y, sr=sr, ax=ax, color='b')
    ax.set_title('Dalga Formu')
    ax.set_xlabel('Zaman (sn)')
    ax.set_ylabel('Genlik')
    return fig

def create_mel_spectrogram(y, sr):
    """
    Create a mel spectrogram visualization
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    # ax parametresi açıkça belirtiliyor
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title('Mel Spektrogramı')
    return fig

def create_chroma(y, sr):
    """
    Create a chromagram visualization
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    # Matplotlib ile doğrudan görselleştirme yaparak hata önleniyor
    img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set_title('Kroma Özellikleri')
    return fig

def main():
    # Dosya yükleme bileşeni
    uploaded_file = st.file_uploader("Analiz etmek istediğiniz müzik dosyasını seçin", type=['mp3', 'wav'])

    if uploaded_file is not None:
        # Dosya bilgilerini göster
        st.write(f"Dosya adı: {uploaded_file.name}")
        st.write(f"Dosya boyutu: {uploaded_file.size} byte")
        
        # Analiz butonu
        if st.button('Analiz Et'):
            # İlerleme durumu için bir yer 
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("Dosya hazırlanıyor...")
            
            # Temp dosya oluştur (Streamlit ile doğrudan dosya yerine geçici kaydedebiliriz)
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                temp_filepath = tmp_file.name
            
            # İlerleme durumu için callback
            def progress_callback(progress):
                progress_bar.progress(progress / 100)
                status_text.text(f"Analiz: %{progress}")
            
            # Analiz işlemini yap
            with st.spinner('Müzik analiz ediliyor...'):
                result = analyze_music(temp_filepath, progress_callback)
            
            # Analiz sonuçlarını göster
            if 'error' in result:
                st.error(f"Analiz sırasında bir hata oluştu: {result['error']}")
            else:
                status_text.text("Analiz tamamlandı!")
                
                # Sonuçları göster
                st.subheader("Analiz Sonuçları")
                
                # Sonuçları tab'lar halinde organize edelim
                tab1, tab2, tab3, tab4 = st.tabs(["Genel Bilgiler", "Tonalite", "Timbre", "Görselleştirmeler"])
                
                with tab1:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Süre", f"{result['duration']:.2f} sn")
                        st.metric("Örnekleme Oranı", f"{result['sample_rate']} Hz")
                    with col2:
                        st.metric("Tempo", f"{result['tempo']:.2f} BPM")
                        st.metric("Ritim Düzeni", result['rhythm_pattern'])
                    
                    st.subheader("Müzik Sistemi")
                    st.info(f"Bu parça {'Batı' if result['tonality']['is_western'] else 'Doğu'} müzik sistemine daha yakın.")
                
                with tab2:
                    st.subheader("Tonalite Analizi")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("Batı Müziği Tonalitesi:")
                        st.info(result['tonality']['western_tonality'])
                        st.progress(result['tonality']['western_confidence'])
                        st.caption(f"Güven: {result['tonality']['western_confidence']:.2f}")
                    
                    with col2:
                        st.write("Doğu Müziği Makamı:")
                        st.info(result['tonality']['eastern_makam'])
                        st.progress(result['tonality']['eastern_confidence'])
                        st.caption(f"Güven: {result['tonality']['eastern_confidence']:.2f}")
                
                with tab3:
                    st.subheader("Timbre (Ses Rengi) Analizi")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Parlaklık", f"{result['timbre']['brightness']:.2f}")
                        st.metric("Zenginlik", f"{result['timbre']['richness']:.2f}")
                    
                    with col2:
                        st.metric("Enstrüman Ailesi", result['timbre']['instrument_family'])
                    
                    # MFCC özelliklerini göster
                    st.subheader("MFCC Özellikleri")
                    st.line_chart(result['timbre']['mfcc_features'])
                
                with tab4:
                    if 'audio_data' in result:
                        # Dalga formunu göster
                        st.subheader("Dalga Formu")
                        y_array = np.array(result['audio_data']['y'])
                        sr = result['audio_data']['sr']
                        
                        waveform_fig = create_waveform(y_array, sr)
                        st.pyplot(waveform_fig)
                        
                        # Spektrogramı göster
                        st.subheader("Mel Spektrogramı")
                        spectrogram_fig = create_mel_spectrogram(y_array, sr)
                        st.pyplot(spectrogram_fig)
                        
                        # Kroma özelliklerini göster
                        st.subheader("Kroma Özellikleri")
                        chroma_fig = create_chroma(y_array, sr)
                        st.pyplot(chroma_fig)
                    else:
                        st.warning("Görselleştirme için gerekli ses verileri eksik.")
                
                # Ham sonuçları JSON olarak göster (geliştiriciler için)
                with st.expander("Ham Analiz Sonuçları (JSON)"):
                    st.json(json.dumps(result, default=custom_serializer))
                
                # Geçici dosyayı temizle
                try:
                    os.unlink(temp_filepath)
                except:
                    pass

    # Yan Panel - Hakkında
    with st.sidebar:
        st.subheader("Hakkında")
        st.write("Bu uygulama, müzik dosyalarını analiz ederek tempo, tonalite, timbre gibi bilgileri çıkarır.")
        st.write("Desteklenen formatlar: .mp3, .wav")
        
        # Ayarlar bölümü
        st.subheader("Ayarlar")
        advanced_mode = st.checkbox("Gelişmiş Mod", value=False)
        
        if advanced_mode:
            st.write("Gelişmiş analiz seçenekleri:")
            analysis_depth = st.slider("Analiz Derinliği", 1, 10, 5)
            st.session_state.analysis_depth = analysis_depth

if __name__ == "__main__":
    main()
