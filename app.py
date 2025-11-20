import streamlit as st
import os
import tempfile
import numpy as np
import librosa
import plotly.graph_objects as go
import json
import pandas as pd
import traceback

# Import configuration
from config import (
    APP_NAME, APP_VERSION, MAX_FILE_SIZE_MB,
    ANALYSIS_DURATION, SUPPORTED_FORMATS,
    ENABLE_DETAILED_ANALYSIS, ENABLE_JSON_EXPORT
)

# Import from analyzer
from analyzer import (
    analyze_music_pure,
    create_frequency_visualization,
    create_koma_analysis_chart,
    create_tonality_comparison_chart,
    create_rhythm_pattern_viz
)

st.set_page_config(
    page_title=f"🎵 {APP_NAME}",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)


def main():
    # Minimal CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown(f'<h1 class="main-header">🎵 {APP_NAME} v{APP_VERSION}</h1>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>Pattern Recognition • Western & Eastern Music Analysis</p>", unsafe_allow_html=True)

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Music File",
        type=SUPPORTED_FORMATS,
        help=f"Supported: {', '.join(SUPPORTED_FORMATS).upper()} • Max {MAX_FILE_SIZE_MB}MB"
    )

    if uploaded_file is not None:
        file_size = len(uploaded_file.getvalue())

        # Check file size
        if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
            st.error(f"❌ File too large! Maximum {MAX_FILE_SIZE_MB}MB allowed.")
            return

        st.success(f"✅ {uploaded_file.name} ({file_size/1024/1024:.1f} MB)")
        st.audio(uploaded_file)

        if st.button("🎵 Analyze Music", type="primary", use_container_width=True):
            with st.spinner("Analyzing..."):
                progress_bar = st.progress(0)
                status_text = st.empty()

                def progress_callback(percent, message=""):
                    progress_bar.progress(percent / 100)
                    status_text.text(f"[{percent:3d}%] {message}")

                try:
                    # Save temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        temp_path = tmp_file.name

                    # Analyze
                    result = analyze_music_pure(temp_path, progress_callback)

                    # Cleanup
                    os.unlink(temp_path)
                    progress_bar.empty()
                    status_text.empty()

                    if 'error' in result:
                        st.error(f"❌ Analysis failed: {result['error']}")
                        return

                    # Success!
                    st.success("✅ Analysis Complete!")

                    # Genre/Style highlight (if available)
                    if 'genre' in result and result['genre']:
                        genre_info = result['genre']
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                                    padding: 1rem; border-radius: 10px; color: white; text-align: center; margin: 1rem 0;">
                            <h3 style="margin: 0;">🎭 {genre_info['primary_genre']}</h3>
                            <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">
                                Confidence: {genre_info['confidence']:.0%}
                                {' • ' + ', '.join(genre_info['all_genres'][:3]) if len(genre_info['all_genres']) > 1 else ''}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                    # Main Results - Compact
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.markdown(f"""
                        <div class="result-card">
                            <h4>System</h4>
                            <h2>{result['tonality']['system']}</h2>
                            <p>{result['tonality']['confidence']:.0%}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        tonality_name = result['tonality']['eastern_makam'] if not result['tonality']['is_western'] else result['tonality']['western_tonality']
                        tonality_type = "Makam" if not result['tonality']['is_western'] else "Scale"
                        st.markdown(f"""
                        <div class="result-card">
                            <h4>{tonality_type}</h4>
                            <h2>{tonality_name}</h2>
                        </div>
                        """, unsafe_allow_html=True)

                    with col3:
                        st.markdown(f"""
                        <div class="result-card">
                            <h4>Tempo</h4>
                            <h2>{result['rhythm']['tempo']:.0f}</h2>
                            <p>{result['rhythm']['meter']} • BPM</p>
                        </div>
                        """, unsafe_allow_html=True)

                    with col4:
                        instruments = ", ".join(result['timbre']['detected_instruments'][:2]) if result['timbre']['detected_instruments'] else "Unknown"
                        st.markdown(f"""
                        <div class="result-card">
                            <h4>Instruments</h4>
                            <h2>{len(result['timbre']['detected_instruments'])}</h2>
                            <p>{instruments}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # Detailed Analysis Tabs (if enabled)
                    if ENABLE_DETAILED_ANALYSIS:
                        st.markdown("---")
                        st.markdown("### 📊 Detailed Analysis")

                        tabs = st.tabs(["Frequency", "Koma", "Comparison", "Rhythm"])

                        with tabs[0]:
                            if result['frequencies']:
                                fig = create_frequency_visualization(result['frequencies'])
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)

                        with tabs[1]:
                            if result['tonality']['koma_analysis']:
                                fig = create_koma_analysis_chart(result['tonality']['koma_analysis'])
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)

                                koma_data = result['tonality']['koma_analysis']
                                microtonal_count = sum(1 for k in koma_data if k['is_microtonal'])

                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total Intervals", len(koma_data))
                                with col2:
                                    st.metric("Microtonal", microtonal_count)
                                with col3:
                                    ratio = (microtonal_count/len(koma_data)*100) if koma_data else 0
                                    st.metric("Ratio", f"{ratio:.1f}%")

                        with tabs[2]:
                            fig = create_tonality_comparison_chart(result['tonality'])
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)

                        with tabs[3]:
                            fig = create_rhythm_pattern_viz(result['rhythm'])
                            st.plotly_chart(fig, use_container_width=True)

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Tempo", f"{result['rhythm']['tempo']:.0f} BPM")
                            with col2:
                                st.metric("Meter", result['rhythm']['meter'])
                            with col3:
                                st.metric("Regularity", f"{result['rhythm']['regularity']:.0%}")

                    # Summary
                    st.markdown("---")
                    st.info(result['summary'])

                    # Technical Details (Collapsible)
                    with st.expander("🔬 Technical Details"):
                        stats = result['analysis_stats']
                        st.json({
                            'Duration (s)': f"{result['duration']:.1f}",
                            'Sample Rate': result['sample_rate'],
                            'Detected Frequencies': stats['total_frequencies'],
                            'Calculated Ratios': stats['total_ratios'],
                            'Microtonal Intervals': stats['microtonal_intervals'],
                            'Microtonal Percentage': f"{result['tonality']['microtonal_ratio']*100:.1f}%",
                            'Western Confidence': f"{result['tonality']['western_confidence']:.1%}",
                            'Eastern Confidence': f"{result['tonality']['eastern_confidence']:.1%}",
                        })

                    # Download (if enabled)
                    if ENABLE_JSON_EXPORT:
                        st.markdown("---")
                        try:
                            result_json = json.dumps(result, indent=2, ensure_ascii=False, default=str)
                            st.download_button(
                                label="📥 Download Results (JSON)",
                                data=result_json,
                                file_name=f"analysis_{uploaded_file.name}.json",
                                mime="application/json",
                                use_container_width=True
                            )
                        except Exception as e:
                            st.warning(f"Export failed: {e}")

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    if st.checkbox("Show error details"):
                        st.code(traceback.format_exc())

    else:
        # Info section (minimal)
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **🎼 Western Music**
            - Major: C, G, D, A, E, F
            - Minor: A, E, B, D, F#

            **🕌 Eastern Music (Makams)**
            - Rast, Hicaz, Nihavend
            - Saba, Hüseyni, Uşşak
            - Segah, Kürdî
            """)

        with col2:
            st.markdown("""
            **🎯 Features**
            - Pattern Recognition Analysis
            - Microtonal Detection (22.64 cent precision)
            - Tempo & Rhythm Analysis
            - Instrument Detection
            - Koma Deviation Analysis
            """)

    # Footer
    st.markdown("---")
    st.markdown(f"<p style='text-align: center; color: #999; font-size: 0.8rem;'>{APP_NAME} v{APP_VERSION} • Pattern Recognition</p>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
