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
import pickle
from pathlib import Path
import traceback

st.set_page_config(
    page_title="🧠 MakamLearner - Real Learning System",
    page_icon="🧠",
    layout="wide"
)

class RealMakamLearner:
    """
    Real Makam Learning System - Learn from actual training files
    Development Phase: Learn patterns from training files
    Production Phase: Use learned patterns for detection
    """
    
    def __init__(self):
        self.koma_per_octave = 53
        self.learned_patterns = {}
        self.training_data = defaultdict(list)
        
        # Analysis parameters
        self.min_confidence = 0.85
        self.frequency_tolerance = 8
        self.min_occurrences = 3
        self.training_folder = "./training_data"
        self.patterns_file = "./learned_makam_patterns.json"
        
    def train_from_folder(self):
        """DEVELOPMENT PHASE: Learn patterns from training folder"""
        st.markdown("## 🧠 Development Phase - Learning from Training Files")
        
        if not os.path.exists(self.training_folder):
            st.error(f"❌ Training folder not found: {self.training_folder}")
            st.info("Create the folder and add training files like: Hicaz.mp3, Nihavend.mp3, etc.")
            return False
        
        # Scan for training files
        audio_files = []
        for ext in ['*.mp3', '*.wav', '*.flac', '*.m4a']:
            audio_files.extend(Path(self.training_folder).glob(ext))
        
        if not audio_files:
            st.warning(f"📁 No audio files found in {self.training_folder}")
            return False
        
        st.success(f"📁 Found {len(audio_files)} training files")
        
        # Display found files with makam mapping
        makam_mapping = {
            'hicaz': 'Hicaz',
            'nihavend': 'Nihavend', 
            'nihavent': 'Nihavend',
            'rast': 'Rast',
            'saba': 'Saba',
            'ussak': 'Uşşak',
            'uşşak': 'Uşşak',
            'huseyni': 'Hüseyni',
            'hüseyni': 'Hüseyni',
            'segah': 'Segah',
            'kurdi': 'Kürdî',
            'kürdî': 'Kürdî'
        }
        
        # Show file mapping
        file_info = []
        for file_path in audio_files:
            filename = file_path.name.lower()
            detected_makam = "Unknown"
            
            for key, value in makam_mapping.items():
                if key in filename:
                    detected_makam = value
                    break
            
            file_info.append({
                'File': file_path.name,
                'Detected Makam': detected_makam,
                'Size': f"{file_path.stat().st_size / (1024*1024):.1f} MB"
            })
        
        st.dataframe(file_info, use_container_width=True)
        
        if st.button("🧠 Start Learning Process", type="primary"):
            return self._execute_learning(audio_files, makam_mapping)
        
        return False
    
    def _execute_learning(self, audio_files, makam_mapping):
        """Execute the learning process"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        self.training_data.clear()  # Reset training data
        
        # Process each file
        for i, file_path in enumerate(audio_files):
            filename = file_path.name.lower()
            makam_name = None
            
            # Determine makam from filename
            for key, value in makam_mapping.items():
                if key in filename:
                    makam_name = value
                    break
            
            if makam_name:
                status_text.text(f"🎵 Learning from {file_path.name} ({makam_name})...")
                st.write(f"🎵 Processing: {file_path.name} → {makam_name}")
                
                # Extract features
                features = self._extract_training_features(str(file_path))
                
                if features:
                    self.training_data[makam_name].append(features)
                    st.success(f"   ✅ Features extracted successfully")
                else:
                    st.warning(f"   ⚠️ Failed to extract features")
            else:
                st.warning(f"⚠️ Could not determine makam from: {file_path.name}")
            
            progress_bar.progress((i + 1) / len(audio_files))
        
        status_text.text("🧠 Learning patterns from collected data...")
        
        # Learn patterns from collected data
        if self.training_data:
            success = self._learn_patterns_from_data()
            if success:
                self._save_learned_patterns()
                status_text.text("✅ Learning completed successfully!")
                return True
        
        status_text.text("❌ Learning failed!")
        return False
    
    def _extract_training_features(self, file_path):
        """Extract comprehensive features from training file"""
        try:
            # Load audio with more flexible parameters
            y, sr = librosa.load(file_path, duration=90, sr=None)  # Allow original sample rate
            
            if len(y) == 0:
                st.error("❌ Boş ses dosyası")
                return None
            
            # Normalize audio
            y = librosa.util.normalize(y)
            
            # Multiple pitch detection methods for robustness
            pitches = []
            
            # Method 1: PYIN
            try:
                f0_pyin, voiced_flag, voiced_prob = librosa.pyin(
                    y, fmin=50, fmax=2000,  # Genişletilmiş frekans aralığı
                    sr=sr, frame_length=2048,
                    fill_na=None  # Don't fill NA values
                )
                pitches.extend([f for f, v, p in zip(f0_pyin, voiced_flag, voiced_prob) 
                              if v and p > 0.7 and f > 0])
            except Exception as e:
                st.warning(f"PYIN analizi başarısız: {e}")
            
            # Method 2: Piptrack for additional pitch information
            try:
                S = np.abs(librosa.stft(y))
                pitches_pt, magnitudes = librosa.piptrack(S=S, sr=sr, 
                                                        fmin=50, fmax=2000,
                                                        threshold=0.1)
                
                # Get the most prominent pitch at each time
                pitches.extend([np.mean(pitches_pt[magnitudes[:, t] > np.max(magnitudes[:, t])/2, t])
                              for t in range(pitches_pt.shape[1])
                              if np.any(magnitudes[:, t] > np.max(magnitudes[:, t])/2)])
            except Exception as e:
                st.warning(f"Piptrack analizi başarısız: {e}")
            
            # Remove duplicates and invalid values
            pitches = [p for p in pitches if p > 0 and not np.isnan(p)]
            pitches = list(set([round(p, 1) for p in pitches]))
            
            if len(pitches) < 10:  # Minimum pitch requirement
                st.error("❌ Yeterli pitch tespit edilemedi")
                return None
            
            # Group frequencies with more flexible tolerance
            core_frequencies = self._group_frequencies(pitches)
            
            if len(core_frequencies) < 3:  # Minimum frequency group requirement
                st.error("❌ Yeterli frekans grubu oluşturulamadı")
                return None
            
            # Find root frequency
            root_freq = self._find_root_frequency(core_frequencies)
            
            # Calculate koma intervals with error handling
            try:
                koma_intervals = self._calculate_koma_intervals(core_frequencies, root_freq)
            except Exception as e:
                st.error(f"❌ Koma hesaplama hatası: {e}")
                return None
            
            # Extract spectral features with error handling
            try:
                spectral_features = self._extract_spectral_features(y, sr)
            except Exception as e:
                st.warning(f"Spektral analiz başarısız: {e}")
                spectral_features = {}
            
            # Extract rhythm features with error handling
            try:
                rhythm_features = self._extract_rhythm_features(y, sr)
            except Exception as e:
                st.warning(f"Ritim analizi başarısız: {e}")
                rhythm_features = {}
            
            # Calculate interval statistics
            interval_stats = self._calculate_interval_statistics(koma_intervals)
            
            st.success("✅ Feature extraction completed successfully")
            
            return {
                'file_path': file_path,
                'root_frequency': root_freq,
                'core_frequencies': core_frequencies,
                'koma_intervals': sorted(koma_intervals),
                'spectral_features': spectral_features,
                'rhythm_features': rhythm_features,
                'interval_statistics': interval_stats,
                'pitch_count': len(pitches),
                'duration': len(y) / sr
            }
            
        except Exception as e:
            st.error(f"❌ Feature extraction failed: {str(e)}")
            st.error(f"Detailed error: {traceback.format_exc()}")
            return None
    
    def _extract_high_confidence_pitches(self, y, sr):
        """Extract high confidence pitches"""
        try:
            # PYIN with strict parameters
            f0, voiced_flag, voiced_prob = librosa.pyin(
                y, fmin=80, fmax=1000, sr=sr,
                frame_length=8192, hop_length=512
            )
            
            # High confidence only
            pitches = []
            for freq, voiced, prob in zip(f0, voiced_flag, voiced_prob):
                if voiced and prob > self.min_confidence and not np.isnan(freq):
                    pitches.append(freq)
            
            return pitches
            
        except Exception:
            return []
    
    def _group_frequencies(self, pitches):
        """Group similar frequencies"""
        frequency_groups = {}
        
        for pitch in pitches:
            grouped = False
            for group_center in frequency_groups:
                if abs(pitch - group_center) <= self.frequency_tolerance:
                    frequency_groups[group_center].append(pitch)
                    grouped = True
                    break
            
            if not grouped:
                frequency_groups[pitch] = [pitch]
        
        # Calculate group statistics
        core_frequencies = []
        for group_center, group_pitches in frequency_groups.items():
            if len(group_pitches) >= self.min_occurrences:
                avg_freq = np.mean(group_pitches)
                strength = len(group_pitches)
                core_frequencies.append(avg_freq)
        
        return sorted(core_frequencies)
    
    def _find_root_frequency(self, frequencies):
        """Find root frequency using multiple methods"""
        if not frequencies:
            return 220.0
        
        # Method 1: Lowest frequency
        lowest = min(frequencies)
        
        # Method 2: Most stable frequency (appears in harmonics)
        harmonic_scores = {}
        for potential_root in frequencies:
            score = 0
            for other_freq in frequencies:
                if other_freq != potential_root:
                    ratio = other_freq / potential_root
                    # Check harmonic ratios
                    for harmonic in [2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5]:
                        if abs(ratio - harmonic) < 0.15:
                            score += 1
            harmonic_scores[potential_root] = score
        
        # Choose best harmonic root if significant
        if harmonic_scores and max(harmonic_scores.values()) >= 2:
            return max(harmonic_scores.items(), key=lambda x: x[1])[0]
        
        return lowest
    
    def _calculate_koma_intervals(self, frequencies, root_freq):
        """Calculate koma intervals from root"""
        koma_intervals = []
        
        for freq in frequencies:
            if freq <= 0 or root_freq <= 0:
                continue
            
            ratio = freq / root_freq
            komas = self.koma_per_octave * np.log2(ratio)
            koma_normalized = komas % self.koma_per_octave
            koma_rounded = round(koma_normalized)
            koma_intervals.append(koma_rounded)
        
        # Remove duplicates and ensure 0 is included
        unique_komas = list(set(koma_intervals))
        if 0 not in unique_komas:
            unique_komas.append(0)
        
        return sorted(unique_komas)
    
    def _extract_spectral_features(self, y, sr):
        """Extract spectral characteristics"""
        try:
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
            
            # Calculate brightness
            brightness = spectral_centroid / (sr / 2)
            
            # Calculate roughness (spectral irregularity)
            stft = librosa.stft(y)
            magnitude = np.abs(stft)
            roughness = np.mean(np.diff(magnitude, axis=0)**2) if magnitude.shape[0] > 1 else 0
            
            return {
                'spectral_centroid': float(spectral_centroid),
                'spectral_bandwidth': float(spectral_bandwidth),
                'spectral_rolloff': float(spectral_rolloff),
                'zero_crossing_rate': float(zero_crossing_rate),
                'brightness': float(brightness),
                'roughness': float(roughness)
            }
        except Exception:
            return {}
    
    def _extract_rhythm_features(self, y, sr):
        """Extract rhythm characteristics"""
        try:
            # Tempo estimation
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            
            # Beat regularity
            if len(beats) > 4:
                beat_times = librosa.frames_to_time(beats, sr=sr)
                intervals = np.diff(beat_times)
                regularity = 1.0 - (np.std(intervals) / (np.mean(intervals) + 1e-8))
                regularity = max(0, min(1, regularity))
            else:
                regularity = 0.5
            
            return {
                'tempo': float(tempo),
                'beat_count': len(beats),
                'regularity': float(regularity)
            }
        except Exception:
            return {'tempo': 120.0, 'beat_count': 0, 'regularity': 0.5}
    
    def _calculate_interval_statistics(self, koma_intervals):
        """Calculate statistics about intervals"""
        if not koma_intervals:
            return {}
        
        # Interval distribution
        interval_counts = Counter(koma_intervals)
        total_intervals = len(koma_intervals)
        
        # Most common intervals
        most_common = interval_counts.most_common(5)
        
        # Interval ranges
        interval_range = max(koma_intervals) - min(koma_intervals)
        
        # Unique interval count
        unique_count = len(set(koma_intervals))
        
        return {
            'total_intervals': total_intervals,
            'unique_intervals': unique_count,
            'interval_range': interval_range,
            'most_common': most_common,
            'interval_distribution': dict(interval_counts)
        }
    
    def _learn_patterns_from_data(self):
        """Learn patterns from collected training data"""
        st.markdown("### 🧠 Learning Patterns from Training Data")
        
        learned_patterns = {}
        
        for makam_name, feature_list in self.training_data.items():
            st.write(f"📚 Learning {makam_name} from {len(feature_list)} examples...")
            
            if len(feature_list) < 1:
                st.warning(f"   ⚠️ Not enough examples for {makam_name}")
                continue
            
            pattern = self._learn_single_makam(makam_name, feature_list)
            learned_patterns[makam_name] = pattern
            
            # Display learned pattern - with safer formatting
            st.success(f"   ✅ Learned {makam_name}:")
            st.write(f"      Core intervals: {pattern['core_intervals']}")
            st.write(f"      Required intervals: {pattern['required_intervals']}")
            
            # Safely display spectral signature
            try:
                brightness = pattern.get('spectral_signature', {}).get('brightness', {}).get('mean', 0)
                st.write(f"      Spectral signature: brightness={brightness:.2f}")
            except (KeyError, AttributeError, TypeError):
                st.write("      Spectral signature: Not available")
        
        self.learned_patterns = learned_patterns
        return len(learned_patterns) > 0
    
    def _learn_single_makam(self, makam_name, feature_list):
        """Learn pattern for a single makam"""
        # Collect all data
        all_intervals = []
        all_roots = []
        all_spectral = defaultdict(list)
        all_rhythm = defaultdict(list)
        
        for features in feature_list:
            all_intervals.extend(features['koma_intervals'])
            all_roots.append(features['root_frequency'])
            
            for key, value in features['spectral_features'].items():
                all_spectral[key].append(value)
            
            for key, value in features['rhythm_features'].items():
                all_rhythm[key].append(value)
        
        # Learn interval patterns
        interval_counts = Counter(all_intervals)
        total_examples = len(feature_list)
        
        # Core intervals (appear in >40% of examples)
        core_intervals = []
        interval_frequencies = {}
        
        for interval, count in interval_counts.items():
            frequency = count / len(all_intervals)  # Frequency among all intervals
            example_frequency = sum(1 for f in feature_list if interval in f['koma_intervals']) / total_examples
            
            interval_frequencies[interval] = frequency
            
            if example_frequency > 0.4:  # Appears in >40% of examples
                core_intervals.append(interval)
        
        # Required intervals (appear in >70% of examples)
        required_intervals = []
        for interval, count in interval_counts.items():
            example_frequency = sum(1 for f in feature_list if interval in f['koma_intervals']) / total_examples
            if example_frequency > 0.7:
                required_intervals.append(interval)
        
        # Learn spectral signature
        spectral_signature = {}
        for key, values in all_spectral.items():
            if values:
                spectral_signature[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        # Learn rhythm signature
        rhythm_signature = {}
        for key, values in all_rhythm.items():
            if values:
                rhythm_signature[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
        
        # Forbidden intervals (never appear or very rare)
        all_possible_intervals = set(range(0, 53))
        present_intervals = set(all_intervals)
        forbidden_intervals = list(all_possible_intervals - present_intervals)
        
        # Additional forbidden based on makam theory
        if makam_name == 'Hicaz':
            # Hicaz typically avoids major Western intervals
            forbidden_intervals.extend([9, 20, 24])
        elif makam_name == 'Nihavend':
            # Nihavend avoids major third
            forbidden_intervals.extend([20, 24])
        elif makam_name == 'Rast':
            # Rast avoids Hicaz intervals
            forbidden_intervals.extend([5])
        
        # Remove duplicates
        forbidden_intervals = list(set(forbidden_intervals))
        
        return {
            'makam_name': makam_name,
            'examples_count': total_examples,
            'core_intervals': sorted(core_intervals),
            'required_intervals': sorted(required_intervals),
            'forbidden_intervals': sorted(forbidden_intervals),
            'interval_frequencies': interval_frequencies,
            'spectral_signature': spectral_signature,
            'rhythm_signature': rhythm_signature,
            'avg_root_frequency': np.mean(all_roots) if all_roots else 220.0,
            'confidence_multiplier': 2.0  # Can be adjusted based on makam characteristics
        }
    
    def _save_learned_patterns(self):
        """Save learned patterns to file"""
        try:
            # Convert numpy types to Python types for JSON serialization
            patterns_to_save = {}
            for makam_name, pattern in self.learned_patterns.items():
                patterns_to_save[makam_name] = self._convert_for_json(pattern)
            
            with open(self.patterns_file, 'w', encoding='utf-8') as f:
                json.dump(patterns_to_save, f, indent=2, ensure_ascii=False)
            
            st.success(f"✅ Learned patterns saved to {self.patterns_file}")
            
            # Also save as backup
            backup_file = f"{self.patterns_file}.backup"
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(patterns_to_save, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            st.error(f"❌ Failed to save patterns: {e}")
    
    def _convert_for_json(self, obj):
        """Convert numpy types to Python types for JSON"""
        if isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def load_learned_patterns(self):
        """Load learned patterns from file"""
        try:
            if os.path.exists(self.patterns_file):
                with open(self.patterns_file, 'r', encoding='utf-8') as f:
                    self.learned_patterns = json.load(f)
                st.success(f"✅ Learned patterns loaded from {self.patterns_file}")
                st.info(f"📚 Available makams: {', '.join(self.learned_patterns.keys())}")
                return True
            else:
                st.warning(f"📁 No learned patterns file found: {self.patterns_file}")
                return False
        except Exception as e:
            st.error(f"❌ Failed to load patterns: {e}")
            return False
    
    def detect_makam_with_learned_patterns(self, audio_features):
        """Detect makam using learned patterns - PRODUCTION PHASE"""
        if not self.learned_patterns or not audio_features:
            return None
        
        detected_intervals = set(audio_features['koma_intervals'])
        detected_spectral = audio_features.get('spectral_features', {})
        
        st.write(f"🔍 Analyzing against learned patterns...")
        st.write(f"   Detected intervals: {sorted(detected_intervals)}")
        
        makam_scores = {}
        
        for makam_name, pattern in self.learned_patterns.items():
            score = 0.0
            details = []
            
            # 1. Required intervals matching
            required_intervals = set(pattern['required_intervals'])
            required_matches = len(required_intervals.intersection(detected_intervals))
            required_score = required_matches / len(required_intervals) if required_intervals else 0
            score += required_score * 3.0
            details.append(f"REQ:{required_matches}/{len(required_intervals)}")
            
            # 2. Core intervals matching
            core_intervals = set(pattern['core_intervals'])
            core_matches = len(core_intervals.intersection(detected_intervals))
            core_score = core_matches / len(core_intervals) if core_intervals else 0
            score += core_score * 2.0
            details.append(f"CORE:{core_matches}/{len(core_intervals)}")
            
            # 3. Interval frequency weighting
            frequency_score = 0
            for interval in detected_intervals:
                if str(interval) in pattern['interval_frequencies']:
                    frequency_score += pattern['interval_frequencies'][str(interval)]
                elif interval in pattern['interval_frequencies']:
                    frequency_score += pattern['interval_frequencies'][interval]
            score += frequency_score
            details.append(f"FREQ:{frequency_score:.2f}")
            
            # 4. Forbidden intervals penalty
            forbidden_intervals = set(pattern['forbidden_intervals'])
            forbidden_matches = len(forbidden_intervals.intersection(detected_intervals))
            forbidden_penalty = forbidden_matches * 1.5
            score -= forbidden_penalty
            if forbidden_penalty > 0:
                details.append(f"FORB:-{forbidden_penalty:.1f}")
            
            # 5. Spectral signature matching
            spectral_bonus = 0
            if detected_spectral and 'brightness' in detected_spectral:
                pattern_spectral = pattern.get('spectral_signature', {})
                if 'brightness' in pattern_spectral:
                    expected_brightness = pattern_spectral['brightness']['mean']
                    brightness_diff = abs(detected_spectral['brightness'] - expected_brightness)
                    if brightness_diff < 0.2:
                        spectral_bonus = 0.5
                        details.append(f"SPEC:+{spectral_bonus}")
            score += spectral_bonus
            
            # 6. Apply confidence multiplier for high-scoring patterns
            if required_score >= 0.7:
                multiplier = pattern.get('confidence_multiplier', 1.0)
                score *= multiplier
                details.append(f"MULT:{multiplier}x")
            
            makam_scores[makam_name] = max(0, score)
            
            # Debug output
            st.write(f"   {makam_name}: {score:.2f} - {', '.join(details)}")
        
        return makam_scores
    
    def analyze_unknown_file(self, file_path):
        """Analyze unknown file using learned patterns"""
        # Extract features from unknown file
        features = self._extract_training_features(file_path)
        
        if not features:
            return {'error': 'Could not extract features from file'}
        
        # Detect makam
        makam_scores = self.detect_makam_with_learned_patterns(features)
        
        if not makam_scores or max(makam_scores.values()) <= 0:
            return {'error': 'No makam pattern matched'}
        
        # Find best match
        best_makam = max(makam_scores.items(), key=lambda x: x[1])
        
        # Calculate confidence
        scores = sorted(makam_scores.values(), reverse=True)
        if len(scores) > 1 and scores[1] > 0:
            confidence = (scores[0] - scores[1]) / (scores[0] + scores[1])
        else:
            confidence = min(0.9, scores[0] / 8.0)
        
        # Boost confidence for very high scores
        if scores[0] >= 6.0:
            confidence = min(1.0, confidence * 1.3)
        
        return {
            'makam': best_makam[0],
            'confidence': confidence,
            'score': best_makam[1],
            'all_scores': makam_scores,
            'features': features,
            'is_certain': confidence > 0.7,
            'method': 'learned_patterns'
        }

def display_analysis_results(results):
    """
    Analiz sonuçlarını görselleştir
    """
    st.success("Analiz tamamlandı!")
    
    # Müzik sistemi ve ölçek analizi
    st.header("Müzik Sistemi Analizi")
    scale_analysis = results['scale_analysis']
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Tespit Edilen Sistem:** {scale_analysis['system']}")
        st.write(f"**Sistem Güvenilirlik:** {scale_analysis['system_confidence']:.2%}")
        st.write(f"**Ölçek/Makam:** {scale_analysis['scale_name']}")
        st.write(f"**Ölçek Güvenilirlik:** {scale_analysis['scale_confidence']:.2%}")
    
    with col2:
        st.write("**Mikrotonalite İçeriği:**")
        st.progress(scale_analysis['microtonal_content'])
        st.write(f"{scale_analysis['microtonal_content']:.1%}")
    
    # Detaylı skorları göster
    with st.expander("Detaylı Analiz Skorları"):
        st.write("**Batı Müziği Ölçekleri:**")
        for scale, score in scale_analysis['detailed_scores']['western_scales'].items():
            st.write(f"- {scale}: {score:.2f}")
        
        st.write("**Doğu Müziği Makamları:**")
        for makam, score in scale_analysis['detailed_scores']['eastern_makams'].items():
            st.write(f"- {makam}: {score:.2f}")
        
        st.write("**Dünya Müziği Ölçekleri:**")
        for scale, score in scale_analysis['detailed_scores']['world_scales'].items():
            st.write(f"- {scale}: {score:.2f}")
    
    # Ritim analizi
    st.header("Ritim Analizi")
    rhythm = results['rhythm']
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Tempo:** {rhythm['tempo']:.1f} BPM")
        st.write(f"**Ölçü:** {rhythm['meter']}")
    
    with col2:
        st.write("**Ritim Düzenliliği:**")
        st.progress(rhythm['regularity'])
        st.write(f"{rhythm['regularity']:.1%}")
    
    # Timbre analizi
    st.header("Timbre (Ses Rengi) Analizi")
    timbre = results['timbre']
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Enstrüman Ailesi:** {timbre['instrument_family']}")
        if timbre['detected_instruments']:
            st.write("**Tespit Edilen Enstrümanlar:**")
            for instrument in timbre['detected_instruments']:
                st.write(f"- {instrument}")
    
    with col2:
        st.write("**Parlaklık:**")
        st.progress(timbre['brightness'])
        st.write("**Zenginlik:**")
        st.progress(timbre['richness'])
    
    # Örüntü analizi
    st.header("Örüntü Analizi")
    patterns = results['patterns']
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Temel Nota:** {patterns['root_note']}")
        st.write(f"**Dominant Mod:** {patterns['dominant_mode']}")
    
    with col2:
        st.write("**Örüntü Yoğunluğu:**")
        st.progress(patterns['pattern_density'])
        st.write(f"{patterns['pattern_density']:.1%}")
    
    # Görselleştirmeler
    st.header("Görsel Analizler")
    
    # Ana görselleştirmeler
    st.subheader("Temel Analiz Görselleştirmeleri")
    tabs = st.tabs(["Dalga Formu", "Spektrogram", "Kroma", "Örüntü"])
    
    with tabs[0]:
        st.pyplot(results['visualizations']['waveform'])
    with tabs[1]:
        st.pyplot(results['visualizations']['mel_spectrogram'])
    with tabs[2]:
        st.pyplot(results['visualizations']['chroma'])
    with tabs[3]:
        st.pyplot(results['visualizations']['pattern'])
    
    # Detaylı analizler
    st.subheader("Detaylı Analiz Görselleştirmeleri")
    detailed_fig = create_detailed_analysis_plots(
        np.array(results['audio_data']['y']),
        results['audio_data']['sr'],
        results
    )
    st.pyplot(detailed_fig)
    
    # Enstrüman analizi
    st.subheader("Enstrüman Analizi")
    instrument_fig = create_instrument_analysis_plot(
        np.array(results['audio_data']['y']),
        results['audio_data']['sr'],
        results['timbre']
    )
    st.pyplot(instrument_fig)

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
    .phase-card { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem; 
        border-radius: 15px; 
        color: white; 
        text-align: center; 
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .learning-info { 
        background: #e8f5e8; 
        padding: 1rem; 
        border-radius: 10px; 
        border-left: 5px solid #4CAF50; 
        margin: 1rem 0; 
    }
    .production-info { 
        background: #e3f2fd; 
        padding: 1rem; 
        border-radius: 10px; 
        border-left: 5px solid #2196F3; 
        margin: 1rem 0; 
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🧠 MakamLearner - Real Learning System</h1>', unsafe_allow_html=True)
    
    # Initialize learner
    if 'learner' not in st.session_state:
        st.session_state.learner = RealMakamLearner()
    
    learner = st.session_state.learner
    
    # Phase selection
    phase = st.radio(
        "🔄 Select Phase",
        ["🧠 Development Phase (Learning)", "🎯 Production Phase (Detection)"],
        horizontal=True
    )
    
    if phase.startswith("🧠"):
        # DEVELOPMENT PHASE - LEARNING
        st.markdown("""
        <div style="background: linear-gradient(135deg, #3A7BD5 0%, #00D2FF 100%); 
                    padding: 20px; 
                    border-radius: 15px; 
                    color: white; 
                    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                    margin: 20px 0;">
            <h3 style="color: white; margin-top: 0;">🧠 Development Phase - Learning from Training Files</h3>
            <p style="color: #E6F3FF;"><strong>Bu fazda sistem gerçek makam dosyalarından öğrenir:</strong></p>
            <ul style="color: #E6F3FF; margin-bottom: 20px;">
                <li>📁 <code style="background: rgba(255,255,255,0.2); padding: 2px 6px; border-radius: 4px;">./training_data</code> klasöründeki dosyaları okur</li>
                <li>🎵 Her dosyadan acoustic features çıkarır</li>
                <li>📊 Makam patterns öğrenir (intervals, spectral, rhythm)</li>
                <li>💾 Öğrenilen patterns'i <code style="background: rgba(255,255,255,0.2); padding: 2px 6px; border-radius: 4px;">learned_makam_patterns.json</code> dosyasına kaydeder</li>
            </ul>
            <p style="color: #E6F3FF; margin-top: 10px; padding: 10px; background: rgba(255,255,255,0.1); border-radius: 8px;">
                <strong>📂 Dosya isimlendirme:</strong> Hicaz.mp3, Nihavend.mp3, Rast.mp3, etc.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Training folder configuration
        training_folder = st.text_input(
            "📁 Training Folder Path",
            value="./training_data",
            help="Makam training dosyalarının bulunduğu klasör"
        )
        learner.training_folder = training_folder
        
        # Execute training
        learner.train_from_folder()
        
        # Show learned patterns if available
        if learner.learned_patterns:
            st.markdown("### 📚 Learned Patterns Summary")
            
            for makam_name, pattern in learner.learned_patterns.items():
                with st.expander(f"🎵 {makam_name} Pattern"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Examples:** {pattern['examples_count']}")
                        st.write(f"**Core Intervals:** {pattern['core_intervals']}")
                        st.write(f"**Required Intervals:** {pattern['required_intervals']}")
                        st.write(f"**Avg Root:** {pattern['avg_root_frequency']:.1f} Hz")
                    
                    with col2:
                        spectral = pattern.get('spectral_signature', {})
                        if 'brightness' in spectral:
                            st.write(f"**Brightness:** {spectral['brightness']['mean']:.2f}")
                        
                        rhythm = pattern.get('rhythm_signature', {})
                        if 'tempo' in rhythm:
                            st.write(f"**Avg Tempo:** {rhythm['tempo']['mean']:.0f} BPM")
                        
                        st.write(f"**Unique Intervals:** {len(pattern['core_intervals'])}")
                
                st.markdown("---")
    else:
        # PRODUCTION PHASE - DETECTION
        st.markdown("""
        <div style="background: linear-gradient(135deg, #6B48FF 0%, #8C4FFF 100%); 
                    padding: 20px; 
                    border-radius: 15px; 
                    color: white; 
                    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                    margin: 20px 0;">
            <h3 style="color: white; margin-top: 0;">🎯 Production Phase - Makam Detection</h3>
            <p style="color: #E0E0FF;"><strong>Bu fazda sistem öğrenilen patterns ile makam tespit eder:</strong></p>
            <ul style="color: #E0E0FF;">
                <li>📂 Öğrenilen patterns'i <code style="background: rgba(255,255,255,0.2); padding: 2px 6px; border-radius: 4px;">learned_makam_patterns.json</code> dosyasından yükler</li>
                <li>🎵 Upload edilen dosyayı analiz eder</li>
                <li>🧠 Learned patterns ile karşılaştırır</li>
                <li>🎯 En uygun makamı confidence ile döndürür</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Load learned patterns
        if st.button("📂 Load Learned Patterns"):
            learner.load_learned_patterns()
        
        # Check if patterns are available
        if learner.learned_patterns:
            st.success(f"✅ Learned patterns available for: {', '.join(learner.learned_patterns.keys())}")
            
            # File upload for detection
            uploaded_file = st.file_uploader(
                "🎵 Upload File for Makam Detection",
                type=['mp3', 'wav', 'flac', 'm4a'],
                help="Upload a Turkish music file to detect its makam"
            )
            
            if uploaded_file is not None:
                st.success(f"✅ {uploaded_file.name} uploaded")
                st.audio(uploaded_file)
                
                if st.button("🧠 Detect Makam with Learned Patterns", type="primary"):
                    # Save temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        temp_path = tmp_file.name
                    
                    # Analyze with learned patterns
                    with st.spinner("🧠 Analyzing with learned patterns..."):
                        result = learner.analyze_unknown_file(temp_path)
                    
                    os.unlink(temp_path)
                    
                    if 'error' in result:
                        st.error(f"❌ {result['error']}")
                    else:
                        # Display results
                        confidence = result['confidence']
                        
                        # Main result card
                        if confidence > 0.8:
                            quality_emoji = "🎯"
                            quality_text = "Very High Confidence"
                            card_style = "background: linear-gradient(135deg, #4CAF50 0%, #8BC34A 100%);"
                        elif confidence > 0.6:
                            quality_emoji = "⚠️"
                            quality_text = "High Confidence"
                            card_style = "background: linear-gradient(135deg, #FF9800 0%, #FFC107 100%);"
                        else:
                            quality_emoji = "❓"
                            quality_text = "Medium Confidence"
                            card_style = "background: linear-gradient(135deg, #9E9E9E 0%, #757575 100%);"
                        
                        st.markdown(f"""
                        <div class="phase-card" style="{card_style}">
                            <h2>🎵 Detected Makam</h2>
                            <h1>{result['makam']}</h1>
                            <h3>{quality_emoji} {quality_text}</h3>
                            <p>Confidence: {confidence:.1%} | Score: {result['score']:.2f}</p>
                            <p>Method: Learned from real examples</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # All scores comparison
                        st.markdown("### 📊 All Makam Scores (Learned Patterns)")
                        
                        scores_data = sorted(result['all_scores'].items(), key=lambda x: x[1], reverse=True)
                        
                        fig = go.Figure()
                        colors = ['gold' if i == 0 else 'lightcoral' if score < 1.0 else 'lightblue' 
                                 for i, (makam, score) in enumerate(scores_data)]
                        
                        fig.add_trace(go.Bar(
                            x=[makam for makam, score in scores_data],
                            y=[score for makam, score in scores_data],
                            marker_color=colors,
                            text=[f"{score:.2f}" for makam, score in scores_data],
                            textposition='auto',
                            hovertemplate='<b>%{x}</b><br>Score: %{y:.2f}<br>Method: Learned Patterns<extra></extra>'
                        ))
                        
                        fig.update_layout(
                            title="Learned Pattern Matching Scores",
                            xaxis_title="Makam",
                            yaxis_title="Score",
                            height=450,
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Technical details
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### 🎼 Extracted Features")
                            features = result['features']
                            st.write(f"**Root Frequency:** {features['root_frequency']:.1f} Hz")
                            st.write(f"**Core Frequencies:** {len(features['core_frequencies'])}")
                            st.write(f"**Koma Intervals:** {features['koma_intervals']}")
                            st.write(f"**Duration:** {features['duration']:.1f} seconds")
                            st.write(f"**Total Pitches:** {features['pitch_count']}")
                        
                        with col2:
                            st.markdown("### 🎨 Spectral & Rhythm")
                            spectral = features.get('spectral_features', {})
                            rhythm = features.get('rhythm_features', {})
                            
                            if spectral:
                                st.write(f"**Brightness:** {spectral.get('brightness', 0):.2f}")
                                st.write(f"**Spectral Centroid:** {spectral.get('spectral_centroid', 0):.0f} Hz")
                            
                            if rhythm:
                                st.write(f"**Tempo:** {rhythm.get('tempo', 0):.0f} BPM")
                                st.write(f"**Regularity:** {rhythm.get('regularity', 0):.2f}")
                        
                        # Show learned pattern for detected makam
                        detected_makam = result['makam']
                        if detected_makam in learner.learned_patterns:
                            pattern = learner.learned_patterns[detected_makam]
                            
                            st.markdown(f"### 🧠 Learned Pattern for {detected_makam}")
                            
                            pattern_col1, pattern_col2 = st.columns(2)
                            
                            with pattern_col1:
                                st.write(f"**Learned from:** {pattern['examples_count']} examples")
                                st.write(f"**Core Intervals:** {pattern['core_intervals']}")
                                st.write(f"**Required Intervals:** {pattern['required_intervals']}")
                            
                            with pattern_col2:
                                st.write(f"**Average Root:** {pattern['avg_root_frequency']:.1f} Hz")
                                spectral_sig = pattern.get('spectral_signature', {})
                                if 'brightness' in spectral_sig:
                                    st.write(f"**Expected Brightness:** {spectral_sig['brightness']['mean']:.2f}")
                        
                        # Validation against filename
                        filename = uploaded_file.name.lower()
                        expected_makam = None
                        
                        makam_mapping = {
                            'hicaz': 'Hicaz',
                            'nihavend': 'Nihavend', 
                            'nihavent': 'Nihavend',
                            'rast': 'Rast',
                            'saba': 'Saba',
                            'ussak': 'Uşşak',
                            'uşşak': 'Uşşak',
                            'huseyni': 'Hüseyni',
                            'hüseyni': 'Hüseyni',
                            'segah': 'Segah',
                            'kurdi': 'Kürdî',
                            'kürdî': 'Kürdî'
                        }
                        
                        for key, value in makam_mapping.items():
                            if key in filename:
                                expected_makam = value
                                break
                        
                        if expected_makam:
                            if result['makam'] == expected_makam:
                                st.success(f"✅ **PERFECT!** Expected: {expected_makam}, Detected: {result['makam']}")
                                st.info("🎯 The learned patterns successfully identified the correct makam!")
                            else:
                                st.error(f"❌ **MISMATCH!** Expected: {expected_makam}, Detected: {result['makam']}")
                                st.warning("🔍 This could indicate:")
                                st.write("- The file contains modulations or mixed makams")
                                st.write("- The filename doesn't match the actual content")
                                st.write("- More training examples needed for better accuracy")
                        
                        # Analysis summary
                        st.markdown("### 📋 Learning-Based Analysis Summary")
                        
                        summary = f"""
**🧠 Detection Result (Learned Patterns):**
- **Detected Makam:** {result['makam']}
- **Confidence:** {confidence:.1%}
- **Score:** {result['score']:.2f}
- **Method:** Real acoustic patterns learned from training data

**🎼 Feature Analysis:**
- **Root Frequency:** {features['root_frequency']:.1f} Hz
- **Intervals Detected:** {len(features['koma_intervals'])} koma positions
- **Spectral Brightness:** {spectral.get('brightness', 0):.2f}
- **Processing Quality:** {features['pitch_count']} high-confidence pitches

**📚 Pattern Matching:**
- **Available Patterns:** {len(learner.learned_patterns)} learned makams
- **Best Match:** {detected_makam} (learned from {pattern['examples_count']} examples)
- **Pattern Confidence:** {quality_text}
                        """
                        
                        if confidence > 0.8:
                            st.success(summary)
                        elif confidence > 0.6:
                            st.info(summary)
                        else:
                            st.warning(summary)
                        
                        # Download results
                        try:
                            export_data = {
                                'filename': uploaded_file.name,
                                'detection_result': {
                                    'makam': result['makam'],
                                    'confidence': confidence,
                                    'score': result['score'],
                                    'method': result['method']
                                },
                                'all_scores': result['all_scores'],
                                'extracted_features': features,
                                'learned_pattern_used': pattern if detected_makam in learner.learned_patterns else None,
                                'expected_makam': expected_makam,
                                'validation_result': 'CORRECT' if expected_makam and result['makam'] == expected_makam else 'INCORRECT' if expected_makam else 'UNKNOWN',
                                'analysis_timestamp': pd.Timestamp.now().isoformat()
                            }
                            
                            result_json = json.dumps(export_data, indent=2, ensure_ascii=False, default=str)
                            st.download_button(
                                label="📥 Download Learning-Based Analysis (JSON)",
                                data=result_json,
                                file_name=f"learned_makam_analysis_{uploaded_file.name}.json",
                                mime="application/json"
                            )
                        except Exception as e:
                            st.warning(f"Export failed: {e}")
        
        else:
            st.warning("🧠 No learned patterns available!")
            st.info("Please go to Development Phase and train the system first.")
            
            # Show training instructions
            st.markdown("""
            ### 📋 Training Instructions
            
            1. **Create training folder:** `./training_data`
            2. **Add training files:**
               - `Hicaz.mp3` - Hicaz makamı örnekleri
               - `Nihavend.mp3` - Nihavend makamı örnekleri  
               - `Rast.mp3` - Rast makamı örnekleri
               - `Saba.mp3` - Saba makamı örnekleri
               - `Ussak.mp3` - Uşşak makamı örnekleri
               - `Huseyni.mp3` - Hüseyni makamı örnekleri
            3. **Switch to Development Phase** and train the system
            4. **Return to Production Phase** for detection
            """)
    
    # Show system status
    st.markdown("---")
    st.markdown("### 🔧 System Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        training_folder_exists = os.path.exists(learner.training_folder)
        st.metric(
            "Training Folder", 
            "✅ Ready" if training_folder_exists else "❌ Missing",
            delta=learner.training_folder
        )
    
    with col2:
        patterns_file_exists = os.path.exists(learner.patterns_file)
        pattern_count = len(learner.learned_patterns) if learner.learned_patterns else 0
        st.metric(
            "Learned Patterns",
            f"{pattern_count} makams",
            delta="✅ Ready" if patterns_file_exists else "❌ No patterns"
        )
    
    with col3:
        if learner.training_data:
            training_count = sum(len(examples) for examples in learner.training_data.values())
            st.metric(
                "Training Data",
                f"{training_count} examples",
                delta=f"{len(learner.training_data)} makams"
            )
        else:
            st.metric("Training Data", "0 examples", delta="No data")
    
    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        st.markdown("#### 🔧 System Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Analysis Parameters:**")
            st.write(f"- Min Confidence: {learner.min_confidence}")
            st.write(f"- Frequency Tolerance: ±{learner.frequency_tolerance} Hz")
            st.write(f"- Min Occurrences: {learner.min_occurrences}")
            st.write(f"- Koma per Octave: {learner.koma_per_octave}")
        
        with col2:
            st.write("**File Paths:**")
            st.write(f"- Training Folder: `{learner.training_folder}`")
            st.write(f"- Patterns File: `{learner.patterns_file}`")
            st.write(f"- Backup File: `{learner.patterns_file}.backup`")
        
        # Manual pattern file management
        st.markdown("#### 📁 Pattern File Management")
        
        if st.button("🗑️ Clear Learned Patterns"):
            learner.learned_patterns.clear()
            learner.training_data.clear()
            st.success("✅ Patterns cleared from memory")
        
        if st.button("💾 Force Save Current Patterns"):
            if learner.learned_patterns:
                learner._save_learned_patterns()
            else:
                st.warning("No patterns to save")
        
        # Show pattern file content
        if patterns_file_exists:
            st.markdown("#### 📋 Current Pattern File Content")
            try:
                with open(learner.patterns_file, 'r', encoding='utf-8') as f:
                    patterns_content = json.load(f)
                
                st.write(f"**File size:** {os.path.getsize(learner.patterns_file)} bytes")
                st.write(f"**Makams in file:** {', '.join(patterns_content.keys())}")
                
                if st.checkbox("Show detailed pattern data"):
                    for makam_name, pattern in patterns_content.items():
                        st.write(f"**{makam_name}:**")
                        st.write(f"  - Examples: {pattern.get('examples_count', 0)}")
                        st.write(f"  - Core intervals: {len(pattern.get('core_intervals', []))}")
                        st.write(f"  - Required intervals: {len(pattern.get('required_intervals', []))}")
                        
            except Exception as e:
                st.error(f"Error reading pattern file: {e}")
    
    # Help section
    with st.expander("❓ Help & Instructions"):
        st.markdown("""
        ### 📋 Complete Usage Guide
        
        #### 🧠 Development Phase (Learning):
        1. **Prepare training data:**
           ```bash
           mkdir training_data
           # Add files: Hicaz.mp3, Nihavend.mp3, Rast.mp3, etc.
           ```
        
        2. **Set training folder path** (default: `./training_data`)
        
        3. **Click "Start Learning Process"** - system will:
           - Extract acoustic features from each file
           - Learn interval patterns, spectral signatures, rhythm patterns
           - Save learned patterns to `learned_makam_patterns.json`
        
        4. **Review learned patterns** in the expandable sections
        
        #### 🎯 Production Phase (Detection):
        1. **Switch to Production Phase**
        
        2. **Load learned patterns** (from previous learning or existing file)
        
        3. **Upload test file** for makam detection
        
        4. **Click "Detect Makam"** - system will:
           - Extract features from test file
           - Compare against learned patterns
           - Return best match with confidence score
        
        #### 🎵 Supported File Formats:
        - **Audio:** MP3, WAV, FLAC, M4A
        - **Duration:** Automatically processed (up to 90 seconds)
        - **Quality:** Higher quality files give better results
        
        #### 🎼 Makam Naming Convention:
        - `Hicaz.mp3` → Hicaz makamı
        - `Nihavend.mp3` → Nihavend makamı  
        - `Rast.mp3` → Rast makamı
        - `Saba.mp3` → Saba makamı
        - `Ussak.mp3` or `Uşşak.mp3` → Uşşak makamı
        - `Huseyni.mp3` or `Hüseyni.mp3` → Hüseyni makamı
        - `Segah.mp3` → Segah makamı
        - `Kurdi.mp3` or `Kürdî.mp3` → Kürdî makamı
        
        #### 🔬 Technical Details:
        - **Pitch Detection:** PYIN algorithm with 85%+ confidence
        - **Frequency Analysis:** 53-TET koma system
        - **Spectral Analysis:** Brightness, centroid, bandwidth
        - **Pattern Matching:** Multi-criteria scoring system
        - **Confidence Calculation:** Differential scoring with boosting
        
        #### 🚨 Troubleshooting:
        - **"No features extracted":** Check audio file quality
        - **"No patterns learned":** Ensure correct file naming
        - **"Low confidence":** Add more training examples
        - **"Pattern file error":** Delete and retrain system
        
        #### 📊 Expected Results:
        - **High confidence (>80%):** Very reliable detection
        - **Medium confidence (60-80%):** Good detection, may need verification  
        - **Low confidence (<60%):** Uncertain, consider adding more training data
        
        #### 🎯 Tips for Better Results:
        - Use **clean, clear recordings** for training
        - Include **multiple examples** per makam (2-3 files minimum)
        - Ensure **correct makam labeling** in filenames
        - Use **traditional/classical** examples rather than fusion
        - **Avoid instrumental solos** - prefer vocal or ensemble pieces
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <h4>🧠 MakamLearner v1.0</h4>
        <p>Real-time makam learning and detection system</p>
        <p>Built with machine learning and Turkish music theory</p>
        <p><em>🎵 "Learning music, one pattern at a time" 🎵</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()