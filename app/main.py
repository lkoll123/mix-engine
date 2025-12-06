# app.py

import sys, os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import List, Optional

import yt_dlp
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent  # project root: mix-engine/
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mix_engine.inference import run_mixer_for_playlist

# ---------------------------
# Create FastAPI app
# ---------------------------

app = FastAPI()

# CORS so your React/Vite frontend can call this API in the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in prod, set to your real domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Config
# ---------------------------

BASE_DIR = Path(__file__).resolve().parent

# songs/ folder next to app.py
SONGS_DIR = BASE_DIR / "songs"
SONGS_DIR.mkdir(parents=True, exist_ok=True)

# mixes/ folder for mixer outputs
MIXES_DIR = BASE_DIR / "mixes"
MIXES_DIR.mkdir(parents=True, exist_ok=True)

# Serve all files under BASE_DIR at /media so frontend can hit wavs directly
app.mount("/media", StaticFiles(directory=str(BASE_DIR)), name="media")


def to_media_url(path: Path) -> str:
    """
    Convert a filesystem path under BASE_DIR into a URL under /media.
    Example:
      BASE_DIR/mixes/<mix_id>/segment_00_*.wav -> /media/mixes/<mix_id>/segment_00_*.wav
    """
    rel = path.relative_to(BASE_DIR)
    return "/media/" + str(rel).replace("\\", "/")


# ---------------------------
# Request/Response Models
# ---------------------------


class PlaylistRequest(BaseModel):
    # This describes the JSON body: { "url": "..." }
    url: str


class ApiResult(BaseModel):
    # This describes the JSON your API returns
    status: int
    message: str
    data: Optional[dict] = None  # can be dict or None


# ---------------------------
# Helper functions
# ---------------------------


def sanitize(name: str) -> str:
    """Rough equivalent of your TS sanitize()."""
    if not name:
        name = "track"
    cleaned = "".join(c if (c.isalnum() or c in (" ", "-", "_")) else "_" for c in name)
    return cleaned[:100]


def convert_to_wav(src_path: Path, out_path: Path):
    """
    Use ffmpeg to convert any input audio file to:
      - WAV
      - 16-bit signed PCM
      - 44.1kHz
      - stereo
    """
    tmp_path = out_path.with_suffix(".wav.tmp")

    cmd = [
        "ffmpeg",
        "-y",  # overwrite
        "-i",
        str(src_path),
        "-acodec",
        "pcm_s16le",
        "-ar",
        "44100",
        "-ac",
        "2",
        "-loglevel",
        "error",
        str(tmp_path),
    ]

    try:
        subprocess.run(cmd, check=True)
        tmp_path.replace(out_path)  # atomic-ish move
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise e


# ---------------------------
# Simple health check route
# ---------------------------


@app.get("/health", response_model=ApiResult)
def health():
    # When you GET /health, this runs.
    return ApiResult(status=200, message="ok")


# ---------------------------
# Main: /getPlaylist (now: download + mix in one step)
# ---------------------------


@app.post("/getPlaylist", response_model=ApiResult)
def get_playlist(req: PlaylistRequest):
    """
    Accepts JSON: { "url": "<soundcloud playlist or track url>" }

    Pipeline:
      1) Download each track via yt-dlp into a temp folder.
      2) Convert each to WAV under ./songs.
      3) Run run_mixer_for_playlist(...) to create seams + segments.
      4) Return URLs for all segments + metadata for frontend to play in order.
    """
    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="No URL provided")

    # temp folder for original downloads before .wav
    tmp_root = SONGS_DIR / f"tmp_{uuid.uuid4().hex}"
    tmp_root.mkdir(parents=True, exist_ok=True)

    files: List[str] = []
    skipped: List[dict] = []
    failures: List[dict] = []

    # base options for yt-dlp
    base_opts = {
        "quiet": True,
        "ignoreerrors": True,
        "noplaylist": False,
    }

    try:
        # First: probe info (no download) to see if it's a playlist / tracks
        with yt_dlp.YoutubeDL({**base_opts, "extract_flat": False}) as ydl:
            info = ydl.extract_info(url, download=False)
            if not info:
                return ApiResult(
                    status=404,
                    message="no_tracks",
                    data={"processed": 0, "skipped": [], "failures": [], "files": []},
                )

        # If playlist -> info["entries"]; if single track -> treat as [info]
        entries = info.get("entries") or [info]

        # Iterate tracks
        for entry in entries:
            if not entry:
                continue

            title = entry.get("title") or f"track_{entry.get('id') or uuid.uuid4().hex}"
            safe_title = sanitize(title)
            wav_out = SONGS_DIR / f"{safe_title}.wav"

            formats = entry.get("formats") or []
            if not formats:
                skipped.append({"title": title, "reason": "no_formats"})
                continue

            try:
                # Set per-track download template in tmp_root
                per_track_opts = {
                    **base_opts,
                    "outtmpl": str(tmp_root / f"{safe_title}.%(ext)s"),
                    "restrictfilenames": True,
                }

                # Download this specific track
                with yt_dlp.YoutubeDL(per_track_opts) as track_ydl:
                    track = track_ydl.extract_info(
                        entry.get("webpage_url") or entry.get("url"),
                        download=True,
                    )
                    downloaded_path = Path(track_ydl.prepare_filename(track))

                # Convert downloaded file -> WAV in SONGS_DIR
                convert_to_wav(downloaded_path, wav_out)

                files.append(str(wav_out))

            except Exception as e:
                failures.append({"title": title, "error": str(e)})
                if wav_out.exists():
                    wav_out.unlink(missing_ok=True)
                continue

    finally:
        # Always clean up temp directory
        if tmp_root.exists():
            shutil.rmtree(tmp_root, ignore_errors=True)

    # if nothing usable
    if not files and not failures and skipped:
        return ApiResult(
            status=404,
            message="no_tracks",
            data={
                "processed": 0,
                "skipped": skipped,
                "failures": failures,
                "files": [],
            },
        )

    # ---------------------------
    # Run full mixer on the downloaded WAVs
    # ---------------------------

    mix_id = uuid.uuid4().hex
    out_dir = MIXES_DIR / mix_id
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        seam_files, seams_meta, segment_files, segment_meta = run_mixer_for_playlist(
            files,
            str(out_dir),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mixing failed: {str(e)}")

    # Map filesystem paths to URLs
    playlist_file_urls = [to_media_url(Path(p)) for p in files]
    seam_file_urls = [to_media_url(Path(p)) for p in seam_files]
    segment_file_urls = [to_media_url(Path(p)) for p in segment_files]

    # success response
    return ApiResult(
        status=200,
        message="success",
        data={
            "mix_id": mix_id,
            "processed": len(files),
            "skipped": skipped,
            "failures": failures,
            "playlist_files": playlist_file_urls,  # original songs as wav
            "seams": {
                "files": seam_file_urls,
                "meta": seams_meta,
            },
            "segments": {
                "files": segment_file_urls,  # this is what you actually play in order
                "meta": segment_meta,
            },
        },
    )
