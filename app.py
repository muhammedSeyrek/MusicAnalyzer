import streamlit as st
import os
import tempfile
import numpy as np
import librosa
import plotly.graph_objects as go
import json
import scipy.signal
from collections import Counter, defaultdict
import pandas as pd
from pathlib import Path
import traceback

# Import from analyzer
from analyzer import (
    analyze_music_pure,
    PureMusicAnalyzer,
    create_frequency_visualization,
    create_koma_analysis_chart,
    create_tonality_comparison_chart,
    create_rhythm_pattern_viz
)

st.set_page_config(
    page_title="🎵 Müzik Analiz Sistemi",
    page_icon="🎵",
    layout="wide"
)


def main():
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .mode-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-header">🎵 Müzik Analiz Sistemi</h1>', unsafe_allow_html=True)
    st.markdown("### Pattern Recognition ile Müzik Analizi")

    st.markdown("""
    <div class="mode-card">
        <h2>🎵 Pattern Recognition Mode</h2>
        <p>Matematiksel analiz ve pattern matching ile tonalite/makam tespiti</p>
        <p><strong>✨ Hiçbir eğitim gerekmez - anında analiz!</strong></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    **Bu sistem şunları yapar:**
    - 🎚️ **Frekans Çıkarımı:** Piptrack + YIN + Chroma yöntemleri
    - 📐 **Koma Sapma Analizi:** 22.64 cent hassasiyetle mikrotonal içerik tespiti
    - 🎼 **Sistem Tespiti:** Batı (Major/Minor) ve Doğu (Makam) müzik sistemleri
    - 🥁 **Ritim Analizi:** Tempo, meter ve aksak ritim tespiti
    - 🎸 **Timbre Analizi:** Enstrüman ve spektral özellik analizi
    - 📊 **Detaylı Görselleştirmeler:** Interaktif grafikler ve raporlar
    """)

    st.markdown("---")

    uploaded_file = st.file_uploader(
        "🎵 Müzik Dosyası Yükleyin",
        type=['mp3', 'wav', 'flac'],
        help="Desteklenen formatlar: MP3, WAV, FLAC (Maksimum 200MB)"
    )

    if uploaded_file is not None:
        file_size = len(uploaded_file.getvalue())
        st.success(f"✅ Dosya yüklendi: {uploaded_file.name} ({file_size/1024/1024:.1f} MB)")
        
        # Audio player
        st.audio(uploaded_file)

        if st.button("🚀 Pattern Recognition Analizi Başlat", type="primary", use_container_width=True):
            with st.spinner("🎵 Pattern recognition analizi yapılıyor..."):
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()

                def progress_callback(percent, message=""):
                    progress_bar.progress(percent / 100)
                    status_text.text(f"[{percent:3d}%] {message}")

                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        temp_path = tmp_file.name

                    # Perform pure analysis
                    result = analyze_music_pure(temp_path, progress_callback)

                    # Clean up
                    os.unlink(temp_path)

                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()

                    if 'error' in result:
                        st.error(f"❌ Analiz hatası: {result['error']}")
                    else:
                        # Display results
                        st.success("🎉 Analiz başarıyla tamamlandı!")

                        # Analysis statistics
                        stats = result['analysis_stats']
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("Tespit Edilen Frekans", stats['total_frequencies'])
                        with col2:
                            st.metric("Hesaplanan Oran", stats['total_ratios'])
                        with col3:
                            st.metric("Mikrotonal Aralık", stats['microtonal_intervals'])
                        with col4:
                            st.metric("Onset Yoğunluğu", f"{stats['onset_density']:.3f}")

                        st.markdown("---")

                        # Main results
                        st.markdown("## 📋 Analiz Sonuçları")

                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.markdown(f"""
                            <div class="mode-card">
                                <h4>🎵 Müzik Sistemi</h4>
                                <h3>{result['tonality']['system']}</h3>
                                <p>Güven: {result['tonality']['confidence']:.1%}</p>
                            </div>
                            """, unsafe_allow_html=True)

                        with col2:
                            tonality_name = result['tonality']['eastern_makam'] if not result['tonality']['is_western'] else result['tonality']['western_tonality']
                            tonality_type = "Makam" if not result['tonality']['is_western'] else "Tonalite"
                            st.markdown(f"""
                            <div class="mode-card">
                                <h4>🎼 {tonality_type}</h4>
                                <h3>{tonality_name}</h3>
                                <p>Matematiksel eşleşme</p>
                            </div>
                            """, unsafe_allow_html=True)

                        with col3:
                            st.markdown(f"""
                            <div class="mode-card">
                                <h4>🥁 Tempo</h4>
                                <h3>{result['rhythm']['tempo']:.0f} BPM</h3>
                                <p>{result['rhythm']['meter']}</p>
                            </div>
                            """, unsafe_allow_html=True)

                        with col4:
                            instrument_count = len(result['timbre']['detected_instruments'])
                            instrument_text = ", ".join(result['timbre']['detected_instruments'][:2]) if result['timbre']['detected_instruments'] else "Belirsiz"
                            st.markdown(f"""
                            <div class="mode-card">
                                <h4>🎸 Enstrümanlar</h4>
                                <h3>{instrument_count}</h3>
                                <p>{instrument_text}</p>
                            </div>
                            """, unsafe_allow_html=True)

                        # Detailed Analysis Tabs
                        st.markdown("## 📊 Detaylı Analiz")

                        tabs = st.tabs(["📈 Frekans Analizi", "🔬 Koma Analizi", "📊 Karşılaştırma", "🥁 Ritim", "🎸 Timbre"])

                        with tabs[0]:
                            st.markdown("### 🎚️ Frekans Analizi")
                            if result['frequencies']:
                                fig = create_frequency_visualization(result['frequencies'])
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)

                                # Show frequencies
                                st.markdown("**Tespit Edilen Frekanslar (Hz):**")
                                freq_cols = st.columns(5)
                                for i, freq in enumerate(result['frequencies'][:20]):
                                    with freq_cols[i % 5]:
                                        st.markdown(f'`{freq:.1f}`')
                            else:
                                st.warning("Frekans verisi bulunamadı")

                        with tabs[1]:
                            st.markdown("### 🔬 Koma Sapma Analizi")
                            if result['tonality']['koma_analysis']:
                                fig = create_koma_analysis_chart(result['tonality']['koma_analysis'])
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)

                                koma_data = result['tonality']['koma_analysis']
                                microtonal_count = sum(1 for k in koma_data if k['is_microtonal'])

                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Toplam Aralık", len(koma_data))
                                with col2:
                                    st.metric("Mikrotonal Aralık", microtonal_count)
                                with col3:
                                    ratio = (microtonal_count/len(koma_data)*100) if koma_data else 0
                                    st.metric("Mikrotonal Oran", f"{ratio:.1f}%")

                                # Detailed koma table
                                with st.expander("🔍 Detaylı Koma Verileri"):
                                    koma_df = pd.DataFrame([
                                        {
                                            'Frekans 1 (Hz)': k['freq_pair'][0],
                                            'Frekans 2 (Hz)': k['freq_pair'][1],
                                            'Oran': f"{k['ratio']:.3f}",
                                            'Koma Sapması': f"{k['koma_deviation']:.2f}",
                                            'Mikrotonal': '✓' if k['is_microtonal'] else '✗'
                                        }
                                        for k in koma_data[:15]
                                    ])
                                    st.dataframe(koma_df, use_container_width=True)
                            else:
                                st.warning("Koma analizi verisi bulunamadı")

                        with tabs[2]:
                            st.markdown("### 📊 Tonalite/Makam Karşılaştırması")
                            fig = create_tonality_comparison_chart(result['tonality'])
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Show scores
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**🎼 Batı Tonaliteleri:**")
                                western_scores = result['tonality'].get('all_western_scores', {})
                                top_western = sorted(western_scores.items(), key=lambda x: x[1], reverse=True)[:5]
                                for scale, score in top_western:
                                    st.write(f"• {scale}: {score:.3f}")
                            
                            with col2:
                                st.markdown("**🕌 Doğu Makamları:**")
                                eastern_scores = result['tonality'].get('all_eastern_scores', {})
                                top_eastern = sorted(eastern_scores.items(), key=lambda x: x[1], reverse=True)[:5]
                                for makam, score in top_eastern:
                                    st.write(f"• {makam}: {score:.3f}")

                        with tabs[3]:
                            st.markdown("### 🥁 Ritim Analizi")
                            fig = create_rhythm_pattern_viz(result['rhythm'])
                            st.plotly_chart(fig, use_container_width=True)

                            # Rhythm details
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Tempo", f"{result['rhythm']['tempo']:.0f} BPM")
                            with col2:
                                st.metric("Ölçü", result['rhythm']['meter'])
                            with col3:
                                st.metric("Düzenlilik", f"{result['rhythm']['regularity']:.1%}")

                            st.markdown("**Ritim Özellikleri:**")
                            st.write(f"• **Karmaşıklık:** {result['rhythm']['complexity']:.2f}")
                            st.write(f"• **Vuruş Sayısı:** {result['rhythm']['beat_count']}")
                            st.write(f"• **Onset Yoğunluğu:** {result['rhythm']['onset_density']:.3f}")

                        with tabs[4]:
                            st.markdown("### 🎸 Timbre ve Enstrüman Analizi")
                            
                            timbre = result['timbre']
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Tespit Edilen Enstrümanlar:**")
                                if timbre['detected_instruments']:
                                    for instrument in timbre['detected_instruments']:
                                        st.write(f"🎶 {instrument.replace('_', ' ').title()}")
                                else:
                                    st.write("❌ Belirgin enstrüman tespit edilemedi")
                            
                            with col2:
                                st.markdown("**Spektral Özellikler:**")
                                st.metric("Parlaklık", f"{timbre['brightness']:.3f}")
                                st.metric("Zenginlik", f"{timbre['richness']:.3f}")
                                st.metric("Harmonik Oran", f"{timbre['harmonic_ratio']:.3f}")
                                st.metric("Perküsif Oran", f"{timbre.get('percussive_ratio', 0):.3f}")

                        # Summary
                        st.markdown("---")
                        st.markdown("## 📝 Analiz Özeti")
                        st.info(result['summary'])

                        # Technical details
                        with st.expander("🔬 Teknik Detaylar ve Metodoloji"):
                            st.markdown("""
                            ### Analiz Metodolojisi:
                            
                            1. **Frekans Çıkarımı:**
                               - Piptrack: Frame-by-frame pitch detection
                               - YIN: Advanced pitch tracking algorithm
                               - Chroma: Pitch class profiling
                            
                            2. **Oran Hesaplama:**
                               - Tüm frekans çiftleri için matematiksel oran hesabı
                               - Oktav içi oranların filtrelenmesi (1.0 - 2.0)
                            
                            3. **Koma Analizi:**
                               - Equal Temperament'tan sapma ölçümü
                               - 1 koma = 22.64 cent hassasiyeti
                               - Mikrotonal içerik tespiti (±0.5 koma eşik)
                            
                            4. **Pattern Matching:**
                               - Matematiksel eşleşme skorları
                               - Karakteristik aralık bonusları
                               - Mikrotonal içerik ağırlıklandırması
                            
                            5. **Karar Verme:**
                               - Objektif skor normalizasyonu
                               - Güven oranı hesaplama
                               - Sistem seçimi (Batı/Doğu)
                            """)
                            
                            st.markdown("### 📊 Analiz İstatistikleri:")
                            st.json({
                                'Toplam Frekans': stats['total_frequencies'],
                                'Toplam Oran': stats['total_ratios'],
                                'Mikrotonal Aralıklar': stats['microtonal_intervals'],
                                'Onset Yoğunluğu': f"{stats['onset_density']:.4f}",
                                'Süre (saniye)': f"{result['duration']:.1f}",
                                'Örnekleme Hızı': result['sample_rate']
                            })

                        # Download results
                        st.markdown("---")
                        try:
                            result_json = json.dumps(result, indent=2, ensure_ascii=False, default=str)
                            st.download_button(
                                label="📥 Tüm Sonuçları JSON Olarak İndir",
                                data=result_json,
                                file_name=f"muzik_analizi_{uploaded_file.name}.json",
                                mime="application/json",
                                use_container_width=True
                            )
                        except Exception as e:
                            st.warning(f"JSON export hatası: {e}")

                except Exception as e:
                    st.error(f"❌ Analiz sırasında hata oluştu: {str(e)}")
                    with st.expander("Hata Detayları"):
                        st.code(traceback.format_exc())
    
    else:
        # Demo/Information section
        st.markdown("## 🎯 Pattern Recognition Prensipleri")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 1️⃣ Saf Frekans Analizi
            
            **🎚️ Üçlü Yaklaşım:**
            - Piptrack (Frame bazlı)
            - YIN algoritması
            - Chroma analizi
            
            **🔍 Filtreleme:**
            - Statistiksel outlier removal
            - IQR bazlı temizleme
            - En belirgin 25 frekans
            """)
        
        with col2:
            st.markdown("""
            ### 2️⃣ Koma Hesabı
            
            **🎵 Hassas Ölçüm:**
            - 22.64 cent koma sistemi
            - Equal Temperament referansı
            - Mikrotonal tespit (±0.5 koma)
            
            **📐 Sapma Analizi:**
            - Cent cinsinden sapma
            - Koma cinsinden sapma
            - İkili sınıflandırma
            """)
        
        with col3:
            st.markdown("""
            ### 3️⃣ Objektif Karar
            
            **🎼 Pattern Matching:**
            - Matematiksel eşleşme
            - Karakteristik aralıklar
            - Mikrotonal bonus
            
            **⚖️ Güven Skoru:**
            - Normalized scoring
            - Confidence calculation
            - No hard-coded rules
            """)

        st.markdown("---")
        st.markdown("""
        ### 📚 Desteklenen Sistemler
        
        **🎼 Batı Müziği:**
        - Major: C, G, D, A, E, F
        - Minor: A, E, B, D, F#
        
        **🕌 Doğu Müziği (Makamlar):**
        - Rast, Hicaz, Nihavend
        - Saba, Hüseyni, Uşşak
        - Segah, Kürdî
        
        **✨ Özellikler:**
        - ✅ Mikrotonal içerik tespiti
        - ✅ Aksak ritim analizi (7/8, 9/8)
        - ✅ Doğu enstrümanları (Ud, Ney, Kanun)
        - ✅ Spektral ve timbre analizi
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <h4>🎵 Müzik Analiz Sistemi v2.0</h4>
        <p>Pattern Recognition ile Müzik Analizi</p>
        <p><em>"Matematiksel hassasiyet, müzikal anlayış"</em></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
