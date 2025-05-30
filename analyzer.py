import streamlit as st
import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import json
import scipy.signal
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(
    page_title="🎵 Dinamik Müzik Analizi",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

class PureMusicAnalyzer:
    """
    Tamamen dinamik müzik analizi sistemi
    Hiçbir şarkı için özel durum yok - sadece gerçek müzikal veriler
    """
    
    def __init__(self):
        # Western scales - Equal Temperament ratios
        self.western_scales = {
            'C Major': [1.000, 1.122, 1.260, 1.335, 1.498, 1.682, 1.888, 2.000],
            'C Minor': [1.000, 1.122, 1.189, 1.335, 1.498, 1.587, 1.782, 2.000],
            'G Major': [1.000, 1.125, 1.265, 1.333, 1.500, 1.687, 1.895, 2.000],
            'D Major': [1.000, 1.120, 1.259, 1.336, 1.496, 1.680, 1.890, 2.000],
            'A Major': [1.000, 1.123, 1.262, 1.334, 1.499, 1.685, 1.886, 2.000],
            'E Major': [1.000, 1.121, 1.258, 1.337, 1.497, 1.683, 1.885, 2.000],
            'F Major': [1.000, 1.122, 1.260, 1.414, 1.498, 1.682, 1.888, 2.000],
            'A Minor': [1.000, 1.122, 1.189, 1.335, 1.498, 1.587, 1.782, 2.000],
            'E Minor': [1.000, 1.120, 1.187, 1.337, 1.497, 1.585, 1.780, 2.000],
            'B Minor': [1.000, 1.123, 1.190, 1.334, 1.499, 1.589, 1.784, 2.000],
            'D Minor': [1.000, 1.122, 1.189, 1.414, 1.498, 1.587, 1.888, 2.000],
            'F# Minor': [1.000, 1.121, 1.190, 1.337, 1.497, 1.589, 1.885, 2.000]
        }
        
        # Eastern makams with koma-based ratios
        self.eastern_makams = {
            'Rast': {
                'ratios': [1.000, 1.125, 1.250, 1.333, 1.500, 1.667, 1.875, 2.000],
                'microtone_positions': [4, 8, 12, 18, 22, 26, 30],  # koma positions
                'characteristic_intervals': [1.125, 1.250, 1.333]
            },
            'Hicaz': {
                'ratios': [1.000, 1.055, 1.125, 1.250, 1.333, 1.500, 1.667, 1.875, 2.000],
                'microtone_positions': [1, 4, 8, 12, 18, 22, 26, 30],
                'characteristic_intervals': [1.055, 1.125, 1.250]
            },
            'Nihavend': {
                'ratios': [1.000, 1.125, 1.200, 1.333, 1.500, 1.600, 1.800, 2.000],
                'microtone_positions': [4, 6, 12, 18, 20, 24, 30],
                'characteristic_intervals': [1.200, 1.333, 1.500]
            },
            'Saba': {
                'ratios': [1.000, 1.055, 1.190, 1.310, 1.420, 1.590, 1.750, 2.000],
                'microtone_positions': [1, 5, 10, 13, 19, 23, 30],
                'characteristic_intervals': [1.055, 1.190, 1.310]
            },
            'Hüseyni': {
                'ratios': [1.000, 1.111, 1.250, 1.350, 1.500, 1.660, 1.800, 2.000],
                'microtone_positions': [3, 8, 11, 18, 21, 24, 30],
                'characteristic_intervals': [1.111, 1.250, 1.350]
            },
            'Uşşak': {
                'ratios': [1.000, 1.111, 1.250, 1.350, 1.500, 1.660, 1.800, 2.000],
                'microtone_positions': [3, 8, 11, 18, 21, 24, 30],
                'characteristic_intervals': [1.111, 1.250, 1.350]
            },
            'Segah': {
                'ratios': [1.000, 1.140, 1.200, 1.320, 1.500, 1.660, 1.780, 2.000],
                'microtone_positions': [4.5, 6, 10.5, 18, 21, 23.5, 30],
                'characteristic_intervals': [1.140, 1.200, 1.320]
            },
            'Kürdî': {
                'ratios': [1.000, 1.111, 1.189, 1.350, 1.500, 1.587, 1.800, 2.000],
                'microtone_positions': [3, 5, 11, 18, 19, 24, 30],
                'characteristic_intervals': [1.111, 1.189, 1.350]
            }
        }

    def extract_frequencies_pure(self, y, sr):
        """
        Saf frekans çıkarımı - hiçbir varsayım yok
        """
        frequencies = []
        
        try:
            # Method 1: Piptrack - frame-by-frame pitch detection
            pitches, magnitudes = librosa.piptrack(
                y=y, sr=sr, threshold=0.1, fmin=80, fmax=2000, 
                hop_length=512, frame_length=2048
            )
            
            # Extract significant pitches
            for t in range(0, pitches.shape[1], 20):  # Sample every 20th frame
                frame_pitches = pitches[:, t]
                frame_magnitudes = magnitudes[:, t]
                
                # Get strongest pitch in this frame
                if np.max(frame_magnitudes) > 0.1:
                    strongest_idx = np.argmax(frame_magnitudes)
                    pitch = frame_pitches[strongest_idx]
                    if pitch > 0:
                        frequencies.append(float(pitch))
            
            # Method 2: YIN algorithm for more accuracy
            try:
                f0_yin = librosa.yin(y, fmin=80, fmax=2000, sr=sr, 
                                   frame_length=2048, hop_length=512)
                # Remove NaN values and add valid frequencies
                valid_yin = f0_yin[~np.isnan(f0_yin)]
                frequencies.extend(valid_yin.tolist())
            except:
                pass
            
            # Method 3: Chroma-based fundamental detection
            try:
                chroma = librosa.feature.chroma_cqt(y=y, sr=sr, bins_per_octave=36)
                
                # Base frequencies (C4 octave)
                base_freqs = [261.63 * (2**(i/12)) for i in range(12)]
                
                for t in range(0, min(chroma.shape[1], 100), 10):
                    chroma_frame = chroma[:, t]
                    # Find dominant pitch classes
                    threshold = np.mean(chroma_frame) + 0.5 * np.std(chroma_frame)
                    dominant_notes = np.where(chroma_frame > threshold)[0]
                    
                    for note_idx in dominant_notes:
                        base_freq = base_freqs[note_idx % 12]
                        # Add multiple octaves
                        frequencies.extend([
                            base_freq * 0.5,  # Lower octave
                            base_freq,        # Base octave
                            base_freq * 2     # Higher octave
                        ])
            except:
                pass
            
            # Clean and filter frequencies
            frequencies = [f for f in frequencies if 80 <= f <= 2000]
            frequencies = list(set([round(f, 1) for f in frequencies]))
            frequencies.sort()
            
            # Remove outliers using statistical method
            if len(frequencies) > 10:
                q1 = np.percentile(frequencies, 25)
                q3 = np.percentile(frequencies, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                frequencies = [f for f in frequencies if lower_bound <= f <= upper_bound]
            
            # Limit to most significant frequencies
            if len(frequencies) > 25:
                frequencies = frequencies[:25]
                
            return frequencies
            
        except Exception as e:
            print(f"Frequency extraction error: {e}")
            return []

    def calculate_frequency_ratios_pure(self, frequencies):
        """
        Saf oran hesaplama - tüm frekans çiftleri
        """
        if len(frequencies) < 2:
            return []
        
        ratios = []
        
        # Calculate all meaningful ratios
        for i in range(len(frequencies)):
            for j in range(i + 1, len(frequencies)):
                f1, f2 = frequencies[i], frequencies[j]
                ratio = f2 / f1
                
                # Only ratios within one octave
                if 1.0 < ratio <= 2.0:
                    ratios.append({
                        'freq1': f1,
                        'freq2': f2,
                        'ratio': ratio,
                        'interval_cents': 1200 * np.log2(ratio)
                    })
        
        return ratios

    def analyze_koma_deviations_pure(self, ratios):
        """
        Gerçek koma analizi - Equal Temperament'tan sapmaları ölç
        """
        koma_analysis = []
        
        for ratio_data in ratios:
            ratio = ratio_data['ratio']
            
            # Find nearest equal temperament semitone
            semitones = 12 * np.log2(ratio)
            nearest_semitone = round(semitones)
            expected_ratio = 2 ** (nearest_semitone / 12)
            
            # Calculate deviations
            cent_deviation = 1200 * np.log2(ratio / expected_ratio)
            koma_deviation = cent_deviation / 22.64  # 1 koma = 22.64 cents
            
            # Classify as microtonal if deviation > 0.5 koma
            is_microtonal = abs(koma_deviation) > 0.5
            
            koma_analysis.append({
                'freq_pair': (float(ratio_data['freq1']), float(ratio_data['freq2'])),
                'ratio': float(ratio),
                'expected_ratio': float(expected_ratio),
                'cent_deviation': float(cent_deviation),
                'koma_deviation': float(koma_deviation),
                'is_microtonal': bool(is_microtonal),
                'semitone_distance': float(nearest_semitone)
            })
        
        return koma_analysis

    def detect_scale_system_pure(self, ratios):
        """
        Saf scale/makam tespiti - sadece matematik
        """
        if not ratios:
            return self._empty_result()
        
        ratio_values = [r['ratio'] for r in ratios]
        
        # Analyze Western scales
        western_scores = {}
        for scale_name, scale_ratios in self.western_scales.items():
            score = self._calculate_mathematical_match(ratio_values, scale_ratios)
            western_scores[scale_name] = score
        
        # Analyze Eastern makams
        eastern_scores = {}
        for makam_name, makam_data in self.eastern_makams.items():
            # Base score from ratio matching
            base_score = self._calculate_mathematical_match(ratio_values, makam_data['ratios'])
            
            # Bonus for characteristic intervals
            char_bonus = self._calculate_characteristic_match(ratio_values, makam_data['characteristic_intervals'])
            
            # Microtonal bonus (Eastern music uses more microtones)
            microtonal_bonus = self._calculate_microtonal_bonus(ratios)
            
            total_score = base_score + char_bonus + microtonal_bonus
            eastern_scores[makam_name] = total_score
        
        # Find best matches
        best_western = max(western_scores.items(), key=lambda x: x[1])
        best_eastern = max(eastern_scores.items(), key=lambda x: x[1])
        
        # Koma analysis for final decision
        koma_analysis = self.analyze_koma_deviations_pure(ratios)
        microtonal_ratio = sum(1 for k in koma_analysis if k['is_microtonal']) / len(koma_analysis) if koma_analysis else 0
        
        # Pure mathematical decision
        # If significant microtonal content (>15%), lean towards Eastern
        # Otherwise, choose based on pure score
        
        eastern_confidence = best_eastern[1]
        western_confidence = best_western[1]
        
        # Microtonal factor
        if microtonal_ratio > 0.15:
            eastern_confidence *= (1 + microtonal_ratio)
        else:
            western_confidence *= (1 + (1 - microtonal_ratio) * 0.2)
        
        # Normalize confidences
        total_conf = eastern_confidence + western_confidence
        if total_conf > 0:
            eastern_confidence /= total_conf
            western_confidence /= total_conf
        
        is_western = western_confidence > eastern_confidence
        
        return {
            'western_tonality': best_western[0],
            'western_confidence': float(western_confidence),
            'eastern_makam': best_eastern[0],
            'eastern_confidence': float(eastern_confidence),
            'is_western': bool(is_western),
            'microtonal_ratio': float(microtonal_ratio),
            'koma_analysis': koma_analysis,
            'system': 'Western' if is_western else 'Eastern',
            'confidence': float(max(western_confidence, eastern_confidence)),
            'all_western_scores': {k: float(v) for k, v in western_scores.items()},
            'all_eastern_scores': {k: float(v) for k, v in eastern_scores.items()}
        }

    def _calculate_mathematical_match(self, observed_ratios, reference_ratios):
        """
        Matematiksel eşleşme skorunu hesapla
        """
        if not observed_ratios or not reference_ratios:
            return 0.0
        
        total_score = 0
        match_count = 0
        
        for obs_ratio in observed_ratios:
            # Find closest reference ratio
            distances = [abs(obs_ratio - ref_ratio) for ref_ratio in reference_ratios]
            min_distance = min(distances)
            
            # Score based on how close the match is (tolerance: 3%)
            if min_distance < 0.03:
                score = (0.03 - min_distance) / 0.03
                total_score += score
                match_count += 1
        
        # Normalize by the number of reference ratios
        return total_score / len(reference_ratios) if reference_ratios else 0.0

    def _calculate_characteristic_match(self, observed_ratios, characteristic_intervals):
        """
        Karakteristik aralık eşleşmesi
        """
        bonus = 0
        for char_interval in characteristic_intervals:
            for obs_ratio in observed_ratios:
                if abs(obs_ratio - char_interval) < 0.02:  # 2% tolerance
                    bonus += 0.1
                    break  # Only count each characteristic once
        return bonus

    def _calculate_microtonal_bonus(self, ratios):
        """
        Mikrotonal içerik bonusu
        """
        if not ratios:
            return 0
        
        microtonal_count = 0
        for ratio_data in ratios:
            ratio = ratio_data['ratio']
            semitones = 12 * np.log2(ratio)
            # Check if ratio is between semitones (microtonal)
            if abs(semitones - round(semitones)) > 0.1:
                microtonal_count += 1
        
        microtonal_ratio = microtonal_count / len(ratios)
        return microtonal_ratio * 0.3  # Bonus for Eastern systems

    def analyze_rhythm_pure(self, y, sr):
        """
        Saf ritim analizi
        """
        try:
            # Onset detection
            onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
            
            # Tempo detection with multiple methods
            tempo_1, beats_1 = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
            tempo_2, beats_2 = librosa.beat.beat_track(y=y, sr=sr)
            
            # Choose the most reasonable tempo
            tempo = tempo_1 if 60 <= tempo_1 <= 200 else tempo_2
            beats = beats_1 if 60 <= tempo_1 <= 200 else beats_2
            
            # Beat regularity analysis
            regularity = 0.5  # Default
            if len(beats) > 4:
                beat_times = librosa.frames_to_time(beats, sr=sr)
                beat_intervals = np.diff(beat_times)
                if len(beat_intervals) > 0:
                    cv = np.std(beat_intervals) / (np.mean(beat_intervals) + 1e-8)
                    regularity = max(0, min(1, 1.0 - cv))
            
            # Meter detection based on onset patterns
            meter = self._detect_meter_mathematically(onset_env, beats)
            
            # Rhythm complexity
            complexity = self._calculate_rhythm_complexity(onset_env)
            
            return {
                'tempo': float(tempo),
                'regularity': float(regularity),
                'meter': meter,
                'beat_count': int(len(beats)),
                'complexity': float(complexity),
                'onset_density': float(np.mean(onset_env))
            }
            
        except Exception as e:
            print(f"Rhythm analysis error: {e}")
            return {
                'tempo': 120.0,
                'regularity': 0.5,
                'meter': '4/4',
                'beat_count': 0,
                'complexity': 0.5,
                'onset_density': 0.5
            }

    def _detect_meter_mathematically(self, onset_env, beats):
        """
        Matematiksel meter tespiti
        """
        if len(beats) < 8:
            return "4/4"
        
        beat_strengths = onset_env[beats[:min(len(beats), 32)]]  # Analyze first 32 beats
        
        # Test different meter patterns using autocorrelation
        patterns = {
            "4/4": 4,
            "3/4": 3,
            "6/8": 6,
            "2/4": 2,
            "7/8": 7,
            "9/8": 9,
            "5/4": 5
        }
        
        scores = {}
        for meter_name, pattern_length in patterns.items():
            if len(beat_strengths) >= pattern_length * 2:
                score = self._calculate_autocorrelation_score(beat_strengths, pattern_length)
                scores[meter_name] = score
        
        # Return meter with highest score, default to 4/4
        return max(scores.items(), key=lambda x: x[1])[0] if scores else "4/4"

    def _calculate_autocorrelation_score(self, signal, period):
        """
        Otomatik korelasyon skoru
        """
        if len(signal) < period * 2:
            return 0
        
        # Calculate autocorrelation at the given period
        correlation = 0
        count = 0
        
        for i in range(len(signal) - period):
            correlation += signal[i] * signal[i + period]
            count += 1
        
        return correlation / count if count > 0 else 0

    def _calculate_rhythm_complexity(self, onset_env):
        """
        Ritim karmaşıklığı hesapla
        """
        # Calculate entropy of onset distribution
        if len(onset_env) == 0:
            return 0.5
        
        # Normalize
        onset_norm = onset_env / (np.max(onset_env) + 1e-8)
        
        # Calculate variance and entropy measures
        variance = np.var(onset_norm)
        mean_onset = np.mean(onset_norm)
        
        complexity = min(1.0, variance + mean_onset)
        return complexity

    def analyze_timbre_pure(self, y, sr):
        """
        Saf timbre analizi - sadece spektral özellikler
        """
        try:
            # Extract spectral features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
            
            # Harmonic/percussive separation
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            harmonic_ratio = np.mean(np.abs(y_harmonic)) / (np.mean(np.abs(y)) + 1e-8)
            percussive_ratio = np.mean(np.abs(y_percussive)) / (np.mean(np.abs(y)) + 1e-8)
            
            # Calculate derived features
            brightness = float(np.mean(spectral_centroid) / (sr/2))
            richness = float(np.mean(spectral_contrast))
            bandwidth = float(np.mean(spectral_bandwidth) / (sr/2))
            rolloff = float(np.mean(spectral_rolloff) / (sr/2))
            zcr = float(np.mean(zero_crossing_rate))
            
            # Instrument classification based on pure spectral analysis
            instruments = self._classify_instruments_mathematically(
                brightness, richness, harmonic_ratio, percussive_ratio, 
                bandwidth, rolloff, zcr
            )
            
            return {
                'brightness': brightness,
                'richness': richness,
                'harmonic_ratio': float(harmonic_ratio),
                'percussive_ratio': float(percussive_ratio),
                'bandwidth': bandwidth,
                'rolloff': rolloff,
                'zero_crossing_rate': zcr,
                'detected_instruments': instruments,
                'mfcc_features': [float(x) for x in mfcc.mean(axis=1)],
                'spectral_features': {
                    'centroid': float(np.mean(spectral_centroid)),
                    'contrast': float(np.mean(spectral_contrast)),
                    'bandwidth': float(np.mean(spectral_bandwidth)),
                    'rolloff': float(np.mean(spectral_rolloff))
                }
            }
            
        except Exception as e:
            print(f"Timbre analysis error: {e}")
            return {
                'brightness': 0.5,
                'richness': 1.0,
                'harmonic_ratio': 0.7,
                'percussive_ratio': 0.3,
                'bandwidth': 0.5,
                'rolloff': 0.5,
                'zero_crossing_rate': 0.1,
                'detected_instruments': [],
                'mfcc_features': [0.0] * 13,
                'spectral_features': {}
            }

    def _classify_instruments_mathematically(self, brightness, richness, harmonic_ratio, 
                                           percussive_ratio, bandwidth, rolloff, zcr):
        """
        Matematiksel enstrüman sınıflandırması
        """
        instruments = []
        
        # String instruments (high harmonic content, moderate brightness)
        if harmonic_ratio > 0.7 and 0.3 < brightness < 0.8:
            if richness > 15:  # Electric instruments have higher contrast
                instruments.append('electric_guitar')
            elif 5 < richness < 15:
                instruments.append('acoustic_guitar')
            elif richness > 8 and brightness < 0.5:
                instruments.append('ud')
            elif brightness > 0.6:
                instruments.append('kanun')
        
        # Piano/keyboard (high harmonic, wide bandwidth, low ZCR)
        if harmonic_ratio > 0.8 and bandwidth > 0.3 and zcr < 0.1:
            instruments.append('piano')
        
        # Wind instruments (moderate harmonic, specific brightness range)
        if 0.6 < harmonic_ratio < 0.8 and 0.4 < brightness < 0.7:
            if rolloff < 0.5:
                instruments.append('ney')
            else:
                instruments.append('flute')
        
        # Percussion (high percussive ratio, high ZCR)
        if percussive_ratio > 0.4 or zcr > 0.15:
            instruments.append('drums')
        
        # Bass instruments (low brightness, high harmonic content)
        if brightness < 0.3 and harmonic_ratio > 0.6:
            instruments.append('bass')
        
        # Brass (high brightness, high richness)
        if brightness > 0.7 and richness > 10:
            instruments.append('brass')
        
        return instruments

    def _empty_result(self):
        """Boş sonuç"""
        return {
            'western_tonality': 'Unknown',
            'western_confidence': 0.0,
            'eastern_makam': 'Unknown',
            'eastern_confidence': 0.0,
            'is_western': True,
            'microtonal_ratio': 0.0,
            'koma_analysis': [],
            'system': 'Unknown',
            'confidence': 0.0,
            'all_western_scores': {},
            'all_eastern_scores': {}
        }

def analyze_music_pure(filepath, progress_callback=None):
    """
    Saf müzik analizi - hiçbir varsayım yok
    """
    analyzer = PureMusicAnalyzer()
    
    if progress_callback:
        progress_callback(10, "Ses dosyası yükleniyor...")
    
    try:
        # Load audio (first 2 minutes for efficiency)
        y, sr = librosa.load(filepath, duration=120)
        
        if progress_callback:
            progress_callback(30, "Frekanslar çıkarılıyor...")
        
        # Extract frequencies
        frequencies = analyzer.extract_frequencies_pure(y, sr)
        
        if progress_callback:
            progress_callback(50, "Frekans oranları hesaplanıyor...")
        
        # Calculate ratios
        ratios = analyzer.calculate_frequency_ratios_pure(frequencies)
        
        if progress_callback:
            progress_callback(70, "Tonalite/makam analizi...")
        
        # Analyze scale system
        tonality = analyzer.detect_scale_system_pure(ratios)
        
        if progress_callback:
            progress_callback(80, "Ritim analizi...")
        
        # Rhythm analysis
        rhythm = analyzer.analyze_rhythm_pure(y, sr)
        
        if progress_callback:
            progress_callback(90, "Timbre analizi...")
        
        # Timbre analysis
        timbre = analyzer.analyze_timbre_pure(y, sr)
        
        if progress_callback:
            progress_callback(100, "Analiz tamamlandı!")
        
        # Create summary
        system_name = tonality['system']
        main_scale = tonality['eastern_makam'] if not tonality['is_western'] else tonality['western_tonality']
        confidence = tonality['confidence']
        
        instruments_str = ', '.join(timbre['detected_instruments']) if timbre['detected_instruments'] else 'Tespit edilemedi'
        
        summary = f"""🎵 Müzik Sistemi: {system_name} (Güven: {confidence:.1%})
🎼 {"Makam" if not tonality['is_western'] else "Tonalite"}: {main_scale}
🥁 Tempo: {rhythm['tempo']:.0f} BPM - {rhythm['meter']}
🎸 Enstrümanlar: {instruments_str}
🔬 Mikrotonal İçerik: %{tonality['microtonal_ratio']*100:.1f}
📊 Ritim Karmaşıklığı: {rhythm['complexity']:.2f}"""
        
        return {
            'duration': float(librosa.get_duration(y=y, sr=sr)),
            'sample_rate': int(sr),
            'frequencies': frequencies,
            'tonality': tonality,
            'rhythm': rhythm,
            'timbre': timbre,
            'summary': summary,
            'analysis_stats': {
                'total_frequencies': len(frequencies),
                'total_ratios': len(ratios),
                'microtonal_intervals': sum(1 for k in tonality['koma_analysis'] if k['is_microtonal']),
                'onset_density': rhythm['onset_density']
            }
        }
        
    except Exception as e:
        print(f"Error analyzing music: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'error': str(e),
            'duration': 0,
            'sample_rate': 44100,
            'frequencies': [],
            'tonality': analyzer._empty_result(),
            'rhythm': {'tempo': 120, 'regularity': 0.5, 'meter': '4/4', 'beat_count': 0, 'complexity': 0.5, 'onset_density': 0.5},
            'timbre': {'brightness': 0, 'richness': 0, 'harmonic_ratio': 0, 'detected_instruments': [], 'mfcc_features': []},
            'summary': 'Analiz başarısız oldu.',
            'analysis_stats': {'total_frequencies': 0, 'total_ratios': 0, 'microtonal_intervals': 0, 'onset_density': 0}
        }

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .result-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .analysis-info {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    
    .frequency-text {
        font-family: 'Courier New', monospace;
        background: #f1f3f4;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

def create_frequency_visualization(frequencies):
    """Frekans görselleştirmesi"""
    if not frequencies:
        return None
        
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(len(frequencies))),
        y=frequencies,
        mode='lines+markers',
        name='Tespit Edilen Frekanslar',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8, color='#ff7f0e')
    ))
    
    fig.update_layout(
        title="Tespit Edilen Temel Frekanslar",
        xaxis_title="Frekans Sırası",
        yaxis_title="Frekans (Hz)",
        template="plotly_white",
        height=400
    )
    
    return fig

def create_koma_analysis_chart(koma_analysis):
    """Koma analizi grafiği"""
    if not koma_analysis:
        return None
    
    deviations = [k['koma_deviation'] for k in koma_analysis]
    microtonal_flags = [k['is_microtonal'] for k in koma_analysis]
    
    colors = ['red' if mt else 'blue' for mt in microtonal_flags]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=list(range(len(deviations))),
        y=deviations,
        marker_color=colors,
        name='Koma Sapmaları',
        text=[f"{d:.2f}" for d in deviations],
        textposition='auto'
    ))
    
    # Mikrotonal eşik çizgileri
    fig.add_hline(y=0.5, line_dash="dash", line_color="green", 
                  annotation_text="Mikrotonal Eşik (+0.5 koma)")
    fig.add_hline(y=-0.5, line_dash="dash", line_color="green", 
                  annotation_text="Mikrotonal Eşik (-0.5 koma)")
    
    fig.update_layout(
        title="Koma Sapma Analizi (Equal Temperament'tan sapmalar)",
        xaxis_title="Aralık No",
        yaxis_title="Koma Sapması",
        template="plotly_white",
        height=400
    )
    
    return fig

def create_tonality_comparison_chart(tonality_data):
    """Tonalite karşılaştırma grafiği"""
    western_scores = tonality_data.get('all_western_scores', {})
    eastern_scores = tonality_data.get('all_eastern_scores', {})
    
    if not western_scores and not eastern_scores:
        return None
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Batı Müziği Tonaliteleri', 'Doğu Müziği Makamları'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Western scores
    if western_scores:
        western_names = list(western_scores.keys())
        western_values = list(western_scores.values())
        
        fig.add_trace(
            go.Bar(x=western_names, y=western_values, name="Western", 
                   marker_color='lightblue'),
            row=1, col=1
        )
    
    # Eastern scores
    if eastern_scores:
        eastern_names = list(eastern_scores.keys())
        eastern_values = list(eastern_scores.values())
        
        fig.add_trace(
            go.Bar(x=eastern_names, y=eastern_values, name="Eastern", 
                   marker_color='lightcoral'),
            row=1, col=2
        )
    
    fig.update_layout(height=400, showlegend=False, title_text="Tonalite/Makam Eşleşme Skorları")
    return fig

def create_rhythm_pattern_viz(rhythm_data):
    """Ritim pattern görselleştirmesi"""
    # Create a simple rhythm visualization
    beats = [1, 0.3, 0.6, 0.3] * 4  # 4/4 pattern repeated
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=list(range(1, len(beats) + 1)),
        y=beats,
        marker_color=['red' if i % 4 == 0 else 'blue' for i in range(len(beats))],
        name='Vuruş Gücü Simülasyonu'
    ))
    
    fig.update_layout(
        title=f"Ritim Paterni - {rhythm_data.get('meter', 'Unknown')} (Tempo: {rhythm_data.get('tempo', 0):.0f} BPM)",
        xaxis_title="Vuruş",
        yaxis_title="Güç",
        template="plotly_white",
        height=300
    )
    
    return fig

def create_waveform(y, sr):
    """
    Ses dalgası görselleştirmesi
    """
    plt.figure(figsize=(10, 4))
    plt.title('Dalga Formu')
    plt.xlabel('Zaman (sn)')
    plt.ylabel('Genlik')
    librosa.display.waveshow(y, sr=sr)
    return plt.gcf()

def create_mel_spectrogram(y, sr):
    """
    Mel spektrogramı görselleştirmesi
    """
    plt.figure(figsize=(10, 4))
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spektrogramı')
    return plt.gcf()

def create_chroma(y, sr):
    """
    Kroma özellikleri görselleştirmesi
    """
    plt.figure(figsize=(10, 4))
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
    plt.colorbar()
    plt.title('Kroma Özellikleri')
    return plt.gcf()

def create_pattern_visualization(y, sr, pattern_period, pattern_density):
    """
    Örüntü analizi görselleştirmesi
    """
    plt.figure(figsize=(12, 8))
    
    # 1. Örüntü yoğunluğu grafiği
    plt.subplot(2, 2, 1)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    rec = librosa.segment.recurrence_matrix(chroma, mode='affinity')
    img = librosa.display.specshow(rec, aspect='equal')
    plt.colorbar()
    plt.title(f'Örüntü Yoğunluğu: {pattern_density:.2f}')
    
    # 2. Tempo ve ritim grafiği
    plt.subplot(2, 2, 2)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    times = librosa.times_like(onset_env, sr=sr)
    plt.plot(times, onset_env)
    plt.title('Ritim Yapısı')
    plt.xlabel('Zaman (sn)')
    plt.ylabel('Vuruş Gücü')
    
    # 3. Harmonik yapı
    plt.subplot(2, 2, 3)
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    S_harmonic = librosa.feature.melspectrogram(y=y_harmonic, sr=sr)
    S_harmonic_db = librosa.power_to_db(S_harmonic, ref=np.max)
    librosa.display.specshow(S_harmonic_db, y_axis='mel', x_axis='time')
    plt.title('Harmonik Yapı')
    plt.colorbar(format='%+2.0f dB')
    
    # 4. Perküsif yapı
    plt.subplot(2, 2, 4)
    S_percussive = librosa.feature.melspectrogram(y=y_percussive, sr=sr)
    S_percussive_db = librosa.power_to_db(S_percussive, ref=np.max)
    librosa.display.specshow(S_percussive_db, y_axis='mel', x_axis='time')
    plt.title('Perküsif Yapı')
    plt.colorbar(format='%+2.0f dB')
    
    plt.tight_layout()
    return plt.gcf()

def create_detailed_analysis_plots(y, sr, analysis_results):
    """
    Detaylı analiz görselleştirmeleri
    """
    plt.figure(figsize=(15, 10))
    
    # 1. Frekans dağılımı
    plt.subplot(2, 2, 1)
    D = np.abs(librosa.stft(y))
    D_db = librosa.amplitude_to_db(D, ref=np.max)
    librosa.display.specshow(D_db, y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Frekans Dağılımı')
    
    # 2. Tonalite/Makam analizi
    plt.subplot(2, 2, 2)
    scale_analysis = analysis_results['scale_analysis']
    system_scores = [
        scale_analysis['detailed_scores']['western_scales'].values(),
        scale_analysis['detailed_scores']['eastern_makams'].values(),
        scale_analysis['detailed_scores']['world_scales'].values()
    ]
    plt.boxplot(system_scores, labels=['Batı', 'Doğu', 'Dünya'])
    plt.title('Müzik Sistemi Karşılaştırması')
    plt.ylabel('Eşleşme Skoru')
    
    # 3. MFCC özellikleri
    plt.subplot(2, 2, 3)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    librosa.display.specshow(mfcc, x_axis='time')
    plt.colorbar()
    plt.title('MFCC Özellikleri')
    
    # 4. Mikrotonalite analizi
    plt.subplot(2, 2, 4)
    cents_deviation = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
    librosa.display.specshow(cents_deviation, y_axis='tonnetz')
    plt.colorbar(label='Cent Sapması')
    plt.title('Mikrotonalite Analizi')
    
    plt.tight_layout()
    return plt.gcf()

def create_instrument_analysis_plot(y, sr, timbre_info):
    """
    Enstrüman analizi görselleştirmesi
    """
    plt.figure(figsize=(12, 6))
    
    # 1. Spektral merkezoid
    plt.subplot(1, 2, 1)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    times = librosa.times_like(cent)
    plt.semilogy(times, cent, label='Spektral Merkezoid')
    plt.ylabel('Frekans (Hz)')
    plt.xlabel('Zaman (s)')
    plt.title('Spektral Özellikler')
    plt.legend()
    
    # 2. Enstrüman karakteristiği
    plt.subplot(1, 2, 2)
    features = ['Parlaklık', 'Zenginlik', 'Harmonik Oran']
    values = [
        timbre_info.get('brightness', 0),
        timbre_info.get('richness', 0),
        timbre_info.get('harmonic_ratio', 0)
    ]
    plt.bar(features, values)
    plt.title('Enstrüman Karakteristiği')
    plt.ylim(0, 1)
    
    plt.tight_layout()
    return plt.gcf()

def main():
    # Header
    st.markdown('<h1 class="main-header">🎵 Dinamik Müzik Analizi Sistemi</h1>', unsafe_allow_html=True)
    st.markdown("### Tamamen Veri Odaklı Doğu ve Batı Müzik Sistemleri Analizi")
    
    # Information about the pure approach
    st.markdown("""
    <div class="analysis-info">
        <h4>🔬 Saf Analiz Yaklaşımı</h4>
        <p>Bu sistem hiçbir şarkı için özel durum yapmaz. Sadece:</p>
        <ul>
            <li>📊 <strong>Matematiksel frekans analizi</strong></li>
            <li>🎵 <strong>Koma sistemi hesaplamaları</strong> (22.64 cent hassasiyetle)</li>
            <li>📐 <strong>Equal Temperament sapma ölçümü</strong></li>
            <li>🔍 <strong>Spektral özellik çıkarımı</strong></li>
            <li>⚖️ <strong>Objektif karar verme algoritmaları</strong></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for analysis settings
    with st.sidebar:
        st.markdown("## ⚙️ Analiz Ayarları")
        
        st.markdown("### 🎯 Tespit Parametreleri")
        show_detailed_analysis = st.checkbox("Detaylı Analiz Göster", value=True)
        show_frequency_analysis = st.checkbox("Frekans Analizi", value=True)
        show_koma_analysis = st.checkbox("Koma Sapma Analizi", value=True)
        show_comparison_charts = st.checkbox("Karşılaştırma Grafikleri", value=True)
        
        st.markdown("---")
        st.markdown("### 📊 Sistem Bilgileri")
        st.markdown("**🎼 Desteklenen Tonaliteler:**")
        st.markdown("• Major: C, G, D, A, E, F")
        st.markdown("• Minor: A, E, B, D, F#")
        
        st.markdown("**🕌 Desteklenen Makamlar:**")
        st.markdown("• Rast, Hicaz, Nihavend")
        st.markdown("• Saba, Hüseyni, Uşşak") 
        st.markdown("• Segah, Kürdî")
        
        st.markdown("---")
        st.markdown("### 🔬 Teknik Detaylar")
        st.markdown("• **Frekans Çıkarımı:** Piptrack + YIN + Chroma")
        st.markdown("• **Koma Hassasiyeti:** 22.64 cent")
        st.markdown("• **Mikrotonal Eşik:** ±0.5 koma")
        st.markdown("• **Analiz Süresi:** İlk 2 dakika")
    
    # File upload
    uploaded_file = st.file_uploader(
        "🎵 Müzik Dosyası Yükleyin",
        type=['mp3', 'wav', 'flac'],
        help="Desteklenen formatlar: MP3, WAV, FLAC (Max: 200MB)"
    )
    
    if uploaded_file is not None:
        # File info
        file_size = len(uploaded_file.getvalue())
        st.success(f"✅ Dosya yüklendi: {uploaded_file.name} ({file_size/1024/1024:.1f} MB)")
        
        # Audio player
        st.audio(uploaded_file, format='audio/mp3')
        
        # Analysis button
        if st.button("🚀 Saf Analizi Başlat", type="primary"):
            with st.spinner("🎵 Dinamik müzik analizi yapılıyor..."):
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
                        return
                    
                    # Display results
                    st.success("🎉 Dinamik analiz başarıyla tamamlandı!")
                    
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
                    
                    # Main results
                    st.markdown("## 📋 Analiz Sonuçları")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>🎵 Müzik Sistemi</h4>
                            <h3>{result['tonality']['system']}</h3>
                            <p>Güven: {result['tonality']['confidence']:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        tonality_name = result['tonality']['eastern_makam'] if not result['tonality']['is_western'] else result['tonality']['western_tonality']
                        tonality_type = "Makam" if not result['tonality']['is_western'] else "Tonalite"
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>🎼 {tonality_type}</h4>
                            <h3>{tonality_name}</h3>
                            <p>Matematiksel eşleşme</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>🥁 Tempo</h4>
                            <h3>{result['rhythm']['tempo']:.0f} BPM</h3>
                            <p>{result['rhythm']['meter']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        instrument_text = ", ".join(result['timbre']['detected_instruments'][:2]) if result['timbre']['detected_instruments'] else "Belirsiz"
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>🎸 Enstrümanlar</h4>
                            <h3>{len(result['timbre']['detected_instruments'])}</h3>
                            <p>{instrument_text}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Detailed Analysis
                    if show_detailed_analysis:
                        st.markdown("## 📊 Detaylı Analiz Sonuçları")
                        
                        # Tonality section
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### 🎼 Tonalite/Makam Bilgileri")
                            tonality = result['tonality']
                            
                            if tonality['is_western']:
                                st.markdown(f"""
                                <div class="result-card">
                                    <h4>🎵 Batı Müziği Sistemi</h4>
                                    <p><strong>{tonality['western_tonality']}</strong></p>
                                    <p><strong>Güven:</strong> {tonality['western_confidence']:.1%}</p>
                                    <p><strong>Mikrotonal İçerik:</strong> {tonality['microtonal_ratio']:.1%}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class="result-card">
                                    <h4>🕌 Doğu Müziği Sistemi</h4>
                                    <p><strong>{tonality['eastern_makam']} Makamı</strong></p>
                                    <p><strong>Güven:</strong> {tonality['eastern_confidence']:.1%}</p>
                                    <p><strong>Mikrotonal İçerik:</strong> {tonality['microtonal_ratio']:.1%}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("### 🥁 Ritim Bilgileri")
                            rhythm = result['rhythm']
                            
                            st.markdown(f"""
                            <div class="result-card">
                                <h4>⏱️ Tempo ve Ritim</h4>
                                <p><strong>Tempo:</strong> {rhythm['tempo']:.0f} BPM</p>
                                <p><strong>Ölçü:</strong> {rhythm['meter']}</p>
                                <p><strong>Düzenlilik:</strong> {rhythm['regularity']:.1%}</p>
                                <p><strong>Karmaşıklık:</strong> {rhythm['complexity']:.2f}</p>
                            </div>
                            """, unsafe_allow_html=True)

                        # Frequency analysis
                        if show_frequency_analysis and result['frequencies']:
                            st.markdown("### 🎚️ Frekans Analizi")
                            freq_fig = create_frequency_visualization(result['frequencies'])
                            if freq_fig:
                                st.plotly_chart(freq_fig, use_container_width=True)
                            
                            # Show frequencies
                            st.markdown("**Tespit Edilen Frekanslar:**")
                            freq_cols = st.columns(5)
                            for i, freq in enumerate(result['frequencies'][:20]):
                                with freq_cols[i % 5]:
                                    st.markdown(f'<span class="frequency-text">{freq:.1f} Hz</span>', unsafe_allow_html=True)
                        
                        # Koma analysis
                        if show_koma_analysis and result['tonality']['koma_analysis']:
                            st.markdown("### 🔬 Koma Analizi")
                            koma_fig = create_koma_analysis_chart(result['tonality']['koma_analysis'])
                            if koma_fig:
                                st.plotly_chart(koma_fig, use_container_width=True)
                            
                            koma_data = result['tonality']['koma_analysis']
                            microtonal_count = sum(1 for k in koma_data if k['is_microtonal'])
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Toplam Aralık", len(koma_data))
                            with col2:
                                st.metric("Mikrotonal Aralık", microtonal_count)
                            with col3:
                                st.metric("Mikrotonal Oran", f"{(microtonal_count/len(koma_data)*100):.1f}%" if koma_data else "0%")

                        # Comparison charts
                        if show_comparison_charts:
                            st.markdown("### 📊 Tonalite/Makam Karşılaştırması")
                            comparison_fig = create_tonality_comparison_chart(result['tonality'])
                            if comparison_fig:
                                st.plotly_chart(comparison_fig, use_container_width=True)
                        
                        # Rhythm visualization
                        st.markdown("### 🎵 Ritim Analizi")
                        rhythm_fig = create_rhythm_pattern_viz(result['rhythm'])
                        st.plotly_chart(rhythm_fig, use_container_width=True)
                        
                        # Timbre analysis
                        st.markdown("### 🎸 Timbre ve Enstrüman Analizi")
                        timbre = result['timbre']
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Tespit Edilen Enstrümanlar:**")
                            if timbre['detected_instruments']:
                                for instrument in timbre['detected_instruments']:
                                    st.markdown(f"🎶 {instrument.title().replace('_', ' ')}")
                            else:
                                st.markdown("❌ Belirgin enstrüman tespit edilemedi")
                        
                        with col2:
                            st.markdown("**Spektral Özellikler:**")
                            st.metric("Parlaklık", f"{timbre['brightness']:.3f}")
                            st.metric("Zenginlik", f"{timbre['richness']:.3f}")
                            st.metric("Harmonik Oran", f"{timbre['harmonic_ratio']:.3f}")
                            st.metric("Perküsif Oran", f"{timbre.get('percussive_ratio', 0):.3f}")
                    
                    # Summary
                    st.markdown("## 📝 Analiz Raporu")
                    st.info(result['summary'])
                    
                    # Technical details expander
                    with st.expander("🔬 Teknik Detaylar"):
                        st.markdown("### Analiz Metodolojisi:")
                        st.markdown("""
                        1. **Frekans Çıkarımı**: Piptrack, YIN ve Chroma tabanlı üçlü yaklaşım
                        2. **Oran Hesaplama**: Tüm frekans çiftleri için matematik hesap
                        3. **Koma Analizi**: Equal Temperament'tan sapma ölçümü (±22.64 cent)
                        4. **Pattern Matching**: Matematiksel eşleşme skorları
                        5. **Karar Verme**: Mikrotonal içerik ağırlıklı objektif algoritma
                        """)
                        
                        if result['tonality']['koma_analysis']:
                            st.markdown("### Koma Sapma Detayları:")
                            koma_df = pd.DataFrame([
                                {
                                    'Frekans 1': k['freq_pair'][0],
                                    'Frekans 2': k['freq_pair'][1], 
                                    'Oran': k['ratio'],
                                    'Koma Sapması': k['koma_deviation'],
                                    'Mikrotonal': k['is_microtonal']
                                } 
                                for k in result['tonality']['koma_analysis'][:10]
                            ])
                            st.dataframe(koma_df)
                    
                    # Download option
                    try:
                        result_json = json.dumps(result, indent=2, ensure_ascii=False, default=str)
                        st.download_button(
                            label="📥 Sonuçları JSON olarak İndir",
                            data=result_json,
                            file_name=f"dinamik_muzik_analizi_{uploaded_file.name}.json",
                            mime="application/json"
                        )
                    except Exception as e:
                        st.warning(f"JSON export hatası: {e}")
                    
                except Exception as e:
                    st.error(f"❌ Analiz sırasında hata oluştu: {str(e)}")
                    with st.expander("Hata Detayları"):
                        import traceback
                        st.code(traceback.format_exc())
    
    else:
        # Demo section
        st.markdown("## 🎯 Dinamik Analiz Prensipleri")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 1️⃣ Saf Frekans Analizi
            🎚️ **Üçlü Yaklaşım:** Piptrack + YIN + Chroma
            
            🔍 **Statistiksel Filtreleme:** Outlier removal
            
            📊 **Objektif Seçim:** En belirgin 25 frekans
            """)
        
        with col2:
            st.markdown("""
            ### 2️⃣ Matematiksel Koma Hesabı
            🎵 **Hassas Ölçüm:** 22.64 cent koma sistemi
            
            📐 **Sapma Analizi:** Equal Temperament referansı
            
            🔬 **Mikrotonal Tespit:** ±0.5 koma eşik
            """)
        
        with col3:
            st.markdown("""
            ### 3️⃣ Objektif Karar Verme
            🎼 **Pattern Matching:** Matematiksel eşleşme
            
            🎹 **Güven Skorları:** Normalizasyonlu skorlama
            
            ⚖️ **Hiçbir Bias Yok:** Sadece veriler konuşur
            """)

if __name__ == "__main__":
    main()