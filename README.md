# Description
Mix-engine is a generative AI system that learns to create smooth, beat-matched, and harmonically compatible transitions between two songs. It combines DSP, deep learning, and music theory to generate DJ-style crossfades and blend points.

# Setup & Run Instructions (Python 3.8 • DJtransGAN • Mix Engine)

## 1. Create and activate Python 3.8 virtual environment
py -3.8 -m venv mixenv  
mixenv\Scripts\activate

## 2. Install project requirements
python -m pip install --upgrade pip  
python -m pip install -r requirements.txt

## 3. Install patched madmom
python -m pip install --no-build-isolation "madmom @ git+https://github.com/CPJKU/madmom.git@v0.16.1"

## 4. Install ffmpeg (required for yt-dlp + audio conversion)
pip install ffmpeg-downloader  
ffdl install --add-path

Restart the terminal and reactivate:
mixenv\Scripts\activate  
ffmpeg -version

## 5. Run the FastAPI backend
uvicorn app.main:app --reload

Server:
http://127.0.0.1:8000  
Docs:
http://127.0.0.1:8000/docs

## 6. Run the full mixing pipeline
POST /getPlaylist with:
{
  "url": "<soundcloud playlist or track url>"
}

Backend will download → convert to WAV → compute transitions → run DJtransGAN → output WAV segments under /media/.


