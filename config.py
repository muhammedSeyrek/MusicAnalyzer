"""
Production Configuration for Music Analyzer
Easily switch between cloud providers
"""

import os

# ============================================
# CLOUD PROVIDER SETTINGS
# ============================================
# Change this to switch providers: 'google', 'aws', 'azure', 'local'
CLOUD_PROVIDER = os.getenv('CLOUD_PROVIDER', 'google')

# ============================================
# APPLICATION SETTINGS
# ============================================
APP_NAME = "Music Analyzer"
APP_VERSION = "2.0"
APP_DESCRIPTION = "Pattern Recognition Music Analysis"

# ============================================
# SERVER CONFIGURATION
# ============================================
# Port configuration - auto-detects cloud provider
if CLOUD_PROVIDER == 'google':
    # Google Cloud Run
    PORT = int(os.getenv('PORT', 8080))
    HOST = '0.0.0.0'
elif CLOUD_PROVIDER == 'aws':
    # AWS Elastic Beanstalk / ECS
    PORT = int(os.getenv('PORT', 8080))
    HOST = '0.0.0.0'
elif CLOUD_PROVIDER == 'azure':
    # Azure App Service
    PORT = int(os.getenv('PORT', 8000))
    HOST = '0.0.0.0'
else:
    # Local development
    PORT = int(os.getenv('PORT', 8501))
    HOST = 'localhost'

# ============================================
# ANALYSIS SETTINGS
# ============================================
# Maximum file size (MB)
MAX_FILE_SIZE_MB = int(os.getenv('MAX_FILE_SIZE_MB', 200))

# Analysis duration (seconds) - use first N seconds of audio
ANALYSIS_DURATION = int(os.getenv('ANALYSIS_DURATION', 120))

# Enable/disable features
ENABLE_DETAILED_ANALYSIS = os.getenv('ENABLE_DETAILED_ANALYSIS', 'true').lower() == 'true'
ENABLE_VISUALIZATIONS = os.getenv('ENABLE_VISUALIZATIONS', 'true').lower() == 'true'
ENABLE_JSON_EXPORT = os.getenv('ENABLE_JSON_EXPORT', 'true').lower() == 'true'

# ============================================
# PERFORMANCE SETTINGS
# ============================================
# Memory limits
MAX_MEMORY_MB = int(os.getenv('MAX_MEMORY_MB', 2048))

# Timeout settings (seconds)
REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', 600))

# ============================================
# LOGGING
# ============================================
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
ENABLE_DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'

# ============================================
# STREAMLIT SPECIFIC
# ============================================
STREAMLIT_CONFIG = {
    'server.port': PORT,
    'server.address': HOST,
    'server.headless': True,
    'server.enableCORS': False,
    'server.enableXsrfProtection': False,
    'browser.gatherUsageStats': False,
    'theme.primaryColor': '#FF6B6B',
    'theme.backgroundColor': '#FFFFFF',
    'theme.secondaryBackgroundColor': '#F0F2F6',
    'theme.textColor': '#262730',
}

# ============================================
# SUPPORTED FORMATS
# ============================================
SUPPORTED_FORMATS = ['mp3', 'wav', 'flac']

# ============================================
# PROVIDER-SPECIFIC OPTIMIZATIONS
# ============================================
if CLOUD_PROVIDER == 'google':
    # Google Cloud Run optimizations
    # Use all available CPU during request
    os.environ['STREAMLIT_SERVER_ENABLE_STATIC_SERVING'] = 'true'

elif CLOUD_PROVIDER == 'aws':
    # AWS Lambda/ECS optimizations
    # Reduce cold start time
    os.environ['PYTHONUNBUFFERED'] = '1'

elif CLOUD_PROVIDER == 'azure':
    # Azure App Service optimizations
    os.environ['WEBSITES_PORT'] = str(PORT)

# ============================================
# HEALTH CHECK
# ============================================
HEALTH_CHECK_PATH = '/_health'

def get_config_summary():
    """Return configuration summary for debugging"""
    return {
        'provider': CLOUD_PROVIDER,
        'port': PORT,
        'host': HOST,
        'max_file_size_mb': MAX_FILE_SIZE_MB,
        'analysis_duration': ANALYSIS_DURATION,
        'debug': ENABLE_DEBUG,
        'version': APP_VERSION
    }
