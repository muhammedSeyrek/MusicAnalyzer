"""
FastAPI Backend for Music Analyzer
Modern REST API for music analysis services
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import tempfile
import os
import traceback
from typing import Optional
import uvicorn

# Import analyzer
from analyzer import analyze_music_pure

app = FastAPI(
    title="Music Analyzer API",
    description="Pattern Recognition Music Analysis API",
    version="2.0.0"
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve React frontend static files
if os.path.exists("frontend/dist"):
    app.mount("/assets", StaticFiles(directory="frontend/dist/assets"), name="assets")

@app.get("/")
async def root():
    """Serve React frontend"""
    if os.path.exists("frontend/dist/index.html"):
        return FileResponse("frontend/dist/index.html")
    return {
        "status": "healthy",
        "service": "Music Analyzer API",
        "version": "2.0.0"
    }

@app.get("/api/health")
async def health():
    """Detailed health check"""
    return {
        "status": "ok",
        "api": "running",
        "analyzer": "ready"
    }

@app.post("/api/analyze")
async def analyze_music(
    file: UploadFile = File(...),
    detailed: Optional[bool] = True
):
    """
    Analyze uploaded music file

    Parameters:
    - file: Audio file (MP3, WAV, FLAC)
    - detailed: Include detailed analysis (default: True)

    Returns:
    - JSON with complete analysis results
    """

    # Validate file type
    allowed_extensions = ['mp3', 'wav', 'flac']
    file_ext = file.filename.split('.')[-1].lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )

    # Check file size (200MB limit)
    max_size = 200 * 1024 * 1024  # 200MB

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_ext}') as tmp_file:
            content = await file.read()

            if len(content) > max_size:
                raise HTTPException(
                    status_code=400,
                    detail="File too large. Maximum 200MB allowed."
                )

            tmp_file.write(content)
            temp_path = tmp_file.name

        # Progress callback for API
        progress_steps = []
        def progress_callback(percent, message=""):
            progress_steps.append({"percent": percent, "message": message})

        # Analyze music
        result = analyze_music_pure(temp_path, progress_callback)

        # Clean up temp file
        os.unlink(temp_path)

        # Check for errors
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])

        # Prepare response
        response = {
            "success": True,
            "filename": file.filename,
            "analysis": {
                "system": result['tonality']['system'],
                "tonality": {
                    "is_western": result['tonality']['is_western'],
                    "western_tonality": result['tonality']['western_tonality'],
                    "eastern_makam": result['tonality']['eastern_makam'],
                    "confidence": result['tonality']['confidence'],
                    "microtonal_ratio": result['tonality']['microtonal_ratio']
                },
                "genre": result.get('genre', {
                    "primary_genre": "Unknown",
                    "confidence": 0,
                    "all_genres": []
                }),
                "rhythm": {
                    "tempo": result['rhythm']['tempo'],
                    "meter": result['rhythm']['meter'],
                    "regularity": result['rhythm']['regularity'],
                    "complexity": result['rhythm']['complexity']
                },
                "timbre": {
                    "instruments": result['timbre']['detected_instruments'],
                    "brightness": result['timbre']['brightness'],
                    "harmonic_ratio": result['timbre']['harmonic_ratio'],
                    "percussive_ratio": result['timbre'].get('percussive_ratio', 0)
                },
                "stats": result['analysis_stats'],
                "duration": result['duration'],
                "sample_rate": result['sample_rate']
            }
        }

        # Include detailed analysis if requested
        if detailed:
            response["analysis"]["detailed"] = {
                "frequencies": result['frequencies'][:50],  # Limit to 50 for performance
                "all_western_scores": result['tonality']['all_western_scores'],
                "all_eastern_scores": result['tonality']['all_eastern_scores'],
                "koma_analysis_count": len(result['tonality']['koma_analysis']),
                "genre_scores": result.get('genre', {}).get('genre_scores', {})
            }

        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        # Clean up temp file if it exists
        try:
            if 'temp_path' in locals():
                os.unlink(temp_path)
        except:
            pass

        # Return error
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )

@app.get("/api/supported-formats")
async def supported_formats():
    """Get list of supported audio formats"""
    return {
        "formats": [
            {"extension": "mp3", "mime": "audio/mpeg"},
            {"extension": "wav", "mime": "audio/wav"},
            {"extension": "flac", "mime": "audio/flac"}
        ],
        "max_size_mb": 200
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        access_log=True
    )
