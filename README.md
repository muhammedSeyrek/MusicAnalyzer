# 🎵 Music Analyzer v2.0

Production-ready music analysis application using pattern recognition to detect tonality, rhythm, and timbre in both Western and Eastern music systems.

## Features

- 🎼 **Western & Eastern Music Detection** - Automatic detection of Major/Minor scales and Turkish Makams
- 📐 **Microtonal Analysis** - 22.64 cent precision koma deviation detection
- 🥁 **Rhythm Analysis** - Tempo, meter, and complex rhythm patterns (including 7/8, 9/8)
- 🎸 **Instrument Detection** - Spectral analysis for instrument classification
- ⚡ **Pattern Recognition** - No training required, instant analysis
- 🌐 **Cloud Ready** - Optimized for Google Cloud Run, AWS, Azure

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

### Cloud Deployment

#### Google Cloud Run (Recommended)

```bash
# One-command deployment
./deploy.sh

# Or manually:
gcloud run deploy music-analyzer \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2
```

#### AWS / Azure

Set environment variable before deploying:

```bash
# For AWS
export CLOUD_PROVIDER=aws

# For Azure
export CLOUD_PROVIDER=azure
```

See `config.py` for provider-specific settings.

## Configuration

All settings are in `config.py`:

```python
# Switch cloud provider
CLOUD_PROVIDER = 'google'  # or 'aws', 'azure', 'local'

# Adjust limits
MAX_FILE_SIZE_MB = 200
ANALYSIS_DURATION = 120  # seconds

# Toggle features
ENABLE_DETAILED_ANALYSIS = True
ENABLE_JSON_EXPORT = True
```

## Supported Music Systems

**Western Music:**
- Major scales: C, G, D, A, E, F
- Minor scales: A, E, B, D, F#

**Eastern Music (Turkish Makams):**
- Rast, Hicaz, Nihavend
- Saba, Hüseyni, Uşşak
- Segah, Kürdî

## Technical Details

### Algorithm

1. **Frequency Extraction**: Piptrack + YIN + Chroma
2. **Ratio Calculation**: All frequency pairs within octave
3. **Koma Analysis**: Equal Temperament deviation measurement
4. **Pattern Matching**: Mathematical score matching
5. **Decision**: Normalized confidence scoring

### Performance

- Analysis time: ~30-60 seconds per song
- Memory usage: ~1.5GB per request
- Supported formats: MP3, WAV, FLAC
- Max file size: 200MB (configurable)

## Requirements

- Python 3.11+
- 2GB RAM recommended
- ffmpeg, libsndfile1 (for audio processing)

See `requirements.txt` for Python dependencies.

## Project Structure

```
music-analyzer/
├── app.py                  # Streamlit application
├── analyzer.py             # Pattern recognition engine
├── config.py               # Configuration (cloud-agnostic)
├── Dockerfile              # Container definition
├── deploy.sh               # Deployment script
├── requirements.txt        # Python dependencies
└── .streamlit/
    └── config.toml         # Streamlit config
```

## Environment Variables

```bash
# Cloud provider
CLOUD_PROVIDER=google

# Port (auto-detected)
PORT=8080

# Features
ENABLE_DETAILED_ANALYSIS=true
ENABLE_JSON_EXPORT=true
MAX_FILE_SIZE_MB=200

# Debug
DEBUG=false
LOG_LEVEL=INFO
```

## Deployment Checklist

- [ ] Update `config.py` with cloud provider
- [ ] Set environment variables
- [ ] Build Docker image or use source deployment
- [ ] Configure memory (2GB) and CPU (2 cores)
- [ ] Set timeout to 600 seconds
- [ ] Enable unauthenticated access (or configure IAM)

## License

MIT License

## Contributing

Issues and pull requests welcome!
