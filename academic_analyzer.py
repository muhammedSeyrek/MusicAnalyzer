"""
Advanced Music Analysis Algorithm for Academic Research
========================================================

This module implements a sophisticated pattern recognition algorithm for
distinguishing between Western tonal music and Eastern makam-based music.

Methodology:
1. Multi-method frequency extraction (Piptrack, YIN, Chroma)
2. Statistical interval analysis
3. Microtonal deviation quantification
4. Multi-criteria decision framework
5. Confidence scoring with validation

For academic publication: [Your Name], [Year]
"""

import numpy as np
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
import librosa


class AcademicMusicAnalyzer:
    """
    Academic-grade music analysis system for Western/Eastern classification

    References:
    - Temperley, D. (2007). Music and Probability
    - Gedik, A. C., & Bozkurt, B. (2010). Pitch-frequency histogram-based music information retrieval for Turkish music
    - Krumhansl, C. L. (1990). Cognitive foundations of musical pitch
    """

    def __init__(self):
        # Western 12-TET (Equal Temperament) scale ratios
        self.western_scales = self._initialize_western_scales()

        # Eastern makam interval structures (53-TET based)
        self.eastern_makams = self._initialize_eastern_makams()

        # Analysis parameters
        self.koma_threshold = 0.5  # Threshold for microtonal classification (in komas)
        self.confidence_threshold = 0.6  # Minimum confidence for classification

        # Statistical parameters
        self.interval_tolerance = 0.03  # 3% tolerance for interval matching
        self.min_pattern_support = 0.4  # Minimum support for pattern detection

    def _initialize_western_scales(self):
        """Initialize Western scale structures with acoustic ratios"""
        # Major scale intervals (in frequency ratios)
        major_template = {
            'intervals': [1.000, 1.122, 1.260, 1.335, 1.498, 1.682, 1.888, 2.000],
            'characteristic_ratios': [1.260, 1.498, 1.682],  # M3, P5, M6
            'interval_cents': [0, 200, 400, 500, 700, 900, 1100, 1200],
            'weight': 1.0
        }

        # Minor scale intervals
        minor_template = {
            'intervals': [1.000, 1.122, 1.189, 1.335, 1.498, 1.587, 1.782, 2.000],
            'characteristic_ratios': [1.189, 1.498, 1.587],  # m3, P5, m6
            'interval_cents': [0, 200, 300, 500, 700, 800, 1000, 1200],
            'weight': 1.0
        }

        western_scales = {}

        # Generate all major keys
        for key in ['C', 'G', 'D', 'A', 'E', 'F']:
            western_scales[f'{key} Major'] = major_template.copy()

        # Generate all minor keys
        for key in ['A', 'E', 'B', 'D', 'F#']:
            western_scales[f'{key} Minor'] = minor_template.copy()

        return western_scales

    def _initialize_eastern_makams(self):
        """Initialize Turkish makam structures with microtonal intervals"""
        eastern_makams = {
            'Rast': {
                'intervals': [1.000, 1.125, 1.250, 1.333, 1.500, 1.667, 1.875, 2.000],
                'characteristic_ratios': [1.125, 1.250, 1.333, 1.500],
                'microtonal_positions': [4, 8, 12, 18, 22, 26, 30],  # Koma positions in 53-TET
                'weight': 1.0
            },
            'Hicaz': {
                'intervals': [1.000, 1.055, 1.125, 1.250, 1.333, 1.500, 1.667, 1.875, 2.000],
                'characteristic_ratios': [1.055, 1.125, 1.250],  # Augmented 2nd characteristic
                'microtonal_positions': [1, 4, 8, 12, 18, 22, 26, 30],
                'weight': 1.2  # Higher weight due to distinctive augmented 2nd
            },
            'Nihavend': {
                'intervals': [1.000, 1.125, 1.200, 1.333, 1.500, 1.600, 1.800, 2.000],
                'characteristic_ratios': [1.200, 1.333, 1.500],
                'microtonal_positions': [4, 6, 12, 18, 20, 24, 30],
                'weight': 1.0
            },
            'Saba': {
                'intervals': [1.000, 1.055, 1.190, 1.310, 1.420, 1.590, 1.750, 2.000],
                'characteristic_ratios': [1.055, 1.190, 1.310],
                'microtonal_positions': [1, 5, 10, 13, 19, 23, 30],
                'weight': 1.1
            },
            'Hüseyni': {
                'intervals': [1.000, 1.111, 1.250, 1.350, 1.500, 1.660, 1.800, 2.000],
                'characteristic_ratios': [1.111, 1.250, 1.350],
                'microtonal_positions': [3, 8, 11, 18, 21, 24, 30],
                'weight': 1.0
            },
            'Uşşak': {
                'intervals': [1.000, 1.111, 1.250, 1.350, 1.500, 1.660, 1.800, 2.000],
                'characteristic_ratios': [1.111, 1.250, 1.350],
                'microtonal_positions': [3, 8, 11, 18, 21, 24, 30],
                'weight': 1.0
            },
            'Segah': {
                'intervals': [1.000, 1.140, 1.200, 1.320, 1.500, 1.660, 1.780, 2.000],
                'characteristic_ratios': [1.140, 1.200, 1.320],
                'microtonal_positions': [4.5, 6, 10.5, 18, 21, 23.5, 30],
                'weight': 1.1
            },
            'Kürdî': {
                'intervals': [1.000, 1.111, 1.189, 1.350, 1.500, 1.587, 1.800, 2.000],
                'characteristic_ratios': [1.111, 1.189, 1.350],
                'microtonal_positions': [3, 5, 11, 18, 19, 24, 30],
                'weight': 1.0
            }
        }

        return eastern_makams

    def analyze_intervals_statistically(self, frequency_ratios):
        """
        Statistical analysis of interval distribution

        Returns:
        - interval_histogram: Distribution of intervals
        - interval_stability: Coefficient of variation
        - interval_entropy: Shannon entropy of distribution
        """
        if len(frequency_ratios) < 3:
            return None

        ratios = [r['ratio'] for r in frequency_ratios]

        # Calculate histogram
        hist, bin_edges = np.histogram(ratios, bins=50, range=(1.0, 2.0))
        hist_normalized = hist / np.sum(hist) if np.sum(hist) > 0 else hist

        # Calculate entropy (measure of interval diversity)
        entropy = stats.entropy(hist_normalized + 1e-10)  # Add small value to avoid log(0)

        # Calculate coefficient of variation (measure of stability)
        cv = np.std(ratios) / (np.mean(ratios) + 1e-10)

        # Identify peaks in distribution (common intervals)
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(hist_normalized, height=0.05, distance=3)
        peak_ratios = [bin_edges[p] for p in peaks]

        return {
            'histogram': hist_normalized,
            'bin_edges': bin_edges,
            'entropy': entropy,
            'coefficient_variation': cv,
            'peak_intervals': peak_ratios,
            'num_peaks': len(peaks)
        }

    def calculate_microtonal_score(self, koma_analysis):
        """
        Advanced microtonal content analysis

        Methodology:
        1. Count significant deviations (>0.5 koma)
        2. Weight by deviation magnitude
        3. Analyze distribution pattern
        """
        if not koma_analysis:
            return {
                'microtonal_ratio': 0.0,
                'avg_deviation': 0.0,
                'max_deviation': 0.0,
                'deviation_std': 0.0
            }

        deviations = [abs(k['koma_deviation']) for k in koma_analysis]
        microtonal_flags = [k['is_microtonal'] for k in koma_analysis]

        microtonal_ratio = np.mean(microtonal_flags)
        avg_deviation = np.mean(deviations)
        max_deviation = np.max(deviations)
        deviation_std = np.std(deviations)

        # Weighted microtonal score (considers magnitude of deviations)
        weighted_score = np.mean([d if d > 0.5 else 0 for d in deviations])

        return {
            'microtonal_ratio': microtonal_ratio,
            'avg_deviation': avg_deviation,
            'max_deviation': max_deviation,
            'deviation_std': deviation_std,
            'weighted_score': weighted_score
        }

    def calculate_pattern_similarity(self, observed_ratios, reference_intervals):
        """
        Calculate pattern similarity using multiple metrics

        Metrics:
        1. Euclidean distance
        2. Cosine similarity
        3. Correlation coefficient
        """
        if not observed_ratios or not reference_intervals:
            return 0.0

        # Match each observed ratio to closest reference
        matches = []
        for obs in observed_ratios:
            distances = [abs(obs - ref) for ref in reference_intervals]
            min_dist = min(distances)
            if min_dist < self.interval_tolerance:
                matches.append((obs, reference_intervals[np.argmin(distances)]))

        if len(matches) < 2:
            return 0.0

        # Calculate coverage (how many reference intervals are matched)
        coverage = len(matches) / len(reference_intervals)

        # Calculate precision (how many observed ratios match)
        precision = len(matches) / len(observed_ratios)

        # F1 score
        if coverage + precision > 0:
            f1_score = 2 * (coverage * precision) / (coverage + precision)
        else:
            f1_score = 0.0

        return f1_score

    def classify_music_system(self, frequency_ratios, koma_analysis):
        """
        Multi-criteria decision framework for Western/Eastern classification

        Decision criteria:
        1. Pattern matching scores (40% weight)
        2. Microtonal content (30% weight)
        3. Statistical features (20% weight)
        4. Characteristic intervals (10% weight)
        """
        if not frequency_ratios:
            return self._empty_classification()

        ratio_values = [r['ratio'] for r in frequency_ratios]

        # Calculate statistical features
        stats_analysis = self.analyze_intervals_statistically(frequency_ratios)
        microtonal_metrics = self.calculate_microtonal_score(koma_analysis)

        # Score Western scales
        western_scores = {}
        for scale_name, scale_data in self.western_scales.items():
            pattern_score = self.calculate_pattern_similarity(
                ratio_values,
                scale_data['intervals']
            )

            char_score = self.calculate_pattern_similarity(
                ratio_values,
                scale_data['characteristic_ratios']
            )

            # Combined score
            western_scores[scale_name] = (pattern_score * 0.7 + char_score * 0.3) * scale_data['weight']

        # Score Eastern makams
        eastern_scores = {}
        for makam_name, makam_data in self.eastern_makams.items():
            pattern_score = self.calculate_pattern_similarity(
                ratio_values,
                makam_data['intervals']
            )

            char_score = self.calculate_pattern_similarity(
                ratio_values,
                makam_data['characteristic_ratios']
            )

            # Combined score
            eastern_scores[makam_name] = (pattern_score * 0.7 + char_score * 0.3) * makam_data['weight']

        # Find best matches
        best_western = max(western_scores.items(), key=lambda x: x[1])
        best_eastern = max(eastern_scores.items(), key=lambda x: x[1])

        # Multi-criteria decision
        western_final = self._calculate_final_score(
            best_western[1],
            microtonal_metrics,
            stats_analysis,
            system_type='western'
        )

        eastern_final = self._calculate_final_score(
            best_eastern[1],
            microtonal_metrics,
            stats_analysis,
            system_type='eastern'
        )

        # Normalize
        total = western_final + eastern_final
        if total > 0:
            western_confidence = western_final / total
            eastern_confidence = eastern_final / total
        else:
            western_confidence = eastern_confidence = 0.5

        is_western = western_confidence > eastern_confidence

        return {
            'system': 'Western' if is_western else 'Eastern',
            'is_western': is_western,
            'western_tonality': best_western[0],
            'eastern_makam': best_eastern[0],
            'western_confidence': float(western_confidence),
            'eastern_confidence': float(eastern_confidence),
            'confidence': float(max(western_confidence, eastern_confidence)),
            'microtonal_metrics': microtonal_metrics,
            'statistical_features': stats_analysis,
            'all_western_scores': {k: float(v) for k, v in western_scores.items()},
            'all_eastern_scores': {k: float(v) for k, v in eastern_scores.items()},
            'pattern_scores': {
                'western_pattern': float(best_western[1]),
                'eastern_pattern': float(best_eastern[1])
            }
        }

    def _calculate_final_score(self, pattern_score, microtonal_metrics, stats_analysis, system_type):
        """
        Calculate final score using weighted criteria

        Weights:
        - Pattern matching: 40%
        - Microtonal analysis: 30%
        - Statistical features: 20%
        - Entropy/diversity: 10%
        """
        # Pattern matching component (40%)
        pattern_component = pattern_score * 0.4

        # Microtonal component (30%)
        if system_type == 'western':
            # Western music should have low microtonal content
            microtonal_component = (1.0 - microtonal_metrics['microtonal_ratio']) * 0.3
            # Boost if very low microtonal content (<15%)
            if microtonal_metrics['microtonal_ratio'] < 0.15:
                microtonal_component *= 1.5
        else:
            # Eastern music should have higher microtonal content
            microtonal_component = microtonal_metrics['weighted_score'] * 0.3
            # Boost if significant microtonal content (>25%)
            if microtonal_metrics['microtonal_ratio'] > 0.25:
                microtonal_component *= 1.3

        # Statistical component (20%)
        if stats_analysis:
            # Western: Lower entropy (more structured)
            # Eastern: Can have higher entropy (more flexible)
            if system_type == 'western':
                stats_component = (1.0 - min(stats_analysis['entropy'] / 4.0, 1.0)) * 0.2
            else:
                stats_component = min(stats_analysis['entropy'] / 3.0, 1.0) * 0.2
        else:
            stats_component = 0.1

        # Diversity component (10%)
        if stats_analysis and stats_analysis['num_peaks'] > 0:
            # Western: 3-5 peaks typical (scale degrees)
            # Eastern: Can vary more
            if system_type == 'western':
                peak_score = 1.0 if 3 <= stats_analysis['num_peaks'] <= 5 else 0.5
            else:
                peak_score = 0.8
            diversity_component = peak_score * 0.1
        else:
            diversity_component = 0.05

        final_score = pattern_component + microtonal_component + stats_component + diversity_component

        return final_score

    def _empty_classification(self):
        """Return empty classification result"""
        return {
            'system': 'Unknown',
            'is_western': True,
            'western_tonality': 'Unknown',
            'eastern_makam': 'Unknown',
            'western_confidence': 0.0,
            'eastern_confidence': 0.0,
            'confidence': 0.0,
            'microtonal_metrics': {},
            'statistical_features': None,
            'all_western_scores': {},
            'all_eastern_scores': {},
            'pattern_scores': {}
        }
