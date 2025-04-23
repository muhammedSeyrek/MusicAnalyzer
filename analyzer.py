import librosa
import numpy as np
import time
from collections import Counter

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
            'is_western': True  # Default to Western when uncertain
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

def analyze_rhythm(y, sr):
    """
    Analyze rhythm patterns and beat structure
    """
    # Get onset strength
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    
    # Get tempo and beats
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    
    # Calculate beat intervals
    if len(beats) > 1:
        beat_intervals = np.diff(beats)
        beat_regularity = 1.0 - np.std(beat_intervals) / np.mean(beat_intervals)
        beat_regularity = max(0, min(1, beat_regularity))  # Normalize to [0,1]
    else:
        beat_regularity = 0
    
    # Determine rhythm pattern
    rhythm_patterns = {
        "4/4": 0,
        "3/4": 0,
        "6/8": 0,
        "5/4": 0,
        "7/8": 0,
        "Complex": 0
    }
    
    # Simple heuristic based on beat strength patterns
    if len(beats) >= 8:
        # Get the strength of each beat
        beat_strengths = onset_env[beats]
        
        # Look at patterns of 4 beats
        for i in range(0, len(beat_strengths) - 4, 4):
            pattern = beat_strengths[i:i+4]
            
            # Analyze relative strengths
            if pattern[0] > pattern[1] and pattern[0] > pattern[2] and pattern[2] > pattern[1]:
                rhythm_patterns["4/4"] += 1
            elif pattern[0] > pattern[1] and pattern[0] > pattern[2]:
                rhythm_patterns["3/4"] += 1
            elif pattern[0] > pattern[1] and pattern[3] > pattern[2]:
                rhythm_patterns["6/8"] += 1
            else:
                rhythm_patterns["Complex"] += 1
    
    # Determine the most likely pattern
    if rhythm_patterns:
        dominant_pattern = max(rhythm_patterns.items(), key=lambda x: x[1])[0]
    else:
        dominant_pattern = "Unknown"
    
    return {
        "tempo": tempo,
        "beat_regularity": beat_regularity,
        "rhythm_pattern": dominant_pattern
    }

def analyze_timbre(y, sr):
    """
    Analyze the timbre characteristics of the music
    """
    # Extract MFCC features for timbre analysis
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Calculate statistics
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_var = np.var(mfcc, axis=1)
    
    # Calculate spectral centroid (brightness)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    brightness = np.mean(spectral_centroid) / (sr/2)  # Normalize by Nyquist frequency
    
    # Calculate spectral contrast (richness)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    richness = np.mean(contrast)
    
    # Determine instrument family (very basic approximation)
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
    
    return {
        "brightness": float(brightness),
        "richness": float(richness),
        "instrument_family": instrument_family,
        "mfcc_features": mfcc_mean.tolist()
    }

def analyze_music(filepath, progress_callback=None):
    """
    Main function to analyze the music file
    """
    # Load the audio file
    try:
        if progress_callback:
            progress_callback(5)  # Starting progress
        
        y, sr = librosa.load(filepath, sr=None)  # Use the file's native sample rate
        
        if progress_callback:
            progress_callback(20)  # File loaded
        
        # Basic audio properties
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Extract pitches
        if progress_callback:
            progress_callback(30)  # Starting pitch extraction
        
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        freqs = []
        for t in range(pitches.shape[1]):
            # Get the frequency with the highest magnitude at this time frame
            if magnitudes[:, t].max() > 0:
                index = magnitudes[:, t].argmax()
                freq = pitches[index, t]
                if freq > 0:  # Only include non-zero frequencies
                    freqs.append(freq)
        
        if progress_callback:
            progress_callback(50)  # Pitches extracted
        
        # Detect tonality
        tonality = detect_tonality(freqs)
        
        if progress_callback:
            progress_callback(70)  # Tonality detected
        
        # Analyze rhythm
        rhythm_info = analyze_rhythm(y, sr)
        
        # Analyze timbre
        timbre_info = analyze_timbre(y, sr)
        
        if progress_callback:
            progress_callback(90)  # Analysis almost complete
        
        # Combine all results
        result = {
            'duration': float(duration),
            'sample_rate': sr,
            'tempo': float(rhythm_info['tempo']),
            'beat_regularity': float(rhythm_info['beat_regularity']),
            'rhythm_pattern': rhythm_info['rhythm_pattern'],
            'tonality': tonality,
            'timbre': timbre_info,
            'frequencies': freqs[:100],  # First 100 frequencies for visualization
            'system': 'Batı' if tonality['is_western'] else 'Doğu'
        }
        
        if progress_callback:
            progress_callback(100)  # Analysis complete
        
        return result
    
    except Exception as e:
        print(f"Error analyzing music: {e}")
        if progress_callback:
            progress_callback(100)  # Ensure progress completes even on error
        
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