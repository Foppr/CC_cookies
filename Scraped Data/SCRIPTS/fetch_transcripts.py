"""Download audio + caption for each TikTok URL in urls.txt and transcribe it.

Writes one JSON per video to transcripts/<id>.json. Already-processed videos are skipped,
so the script can be re-run after interruptions or rate limits.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import yt_dlp
import numpy as np
from faster_whisper import WhisperModel

ROOT = Path(__file__).parent
AUDIO = ROOT / "audio"
OUT = ROOT / "transcripts"
AUDIO.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

urls = [u.strip() for u in (ROOT / "urls.txt").read_text().splitlines() if u.strip()]
model = WhisperModel("small", device="cpu", compute_type="int8")


def load_audio(path):
    # Decode with ffmpeg rather than PyAV: av 19 is incompatible with faster-whisper 1.2.1.
    pcm = subprocess.run(
        ["ffmpeg", "-nostdin", "-i", str(path), "-f", "s16le", "-ac", "1", "-ar", "16000", "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0


ydl_opts = {
    "format": "bestaudio/best",
    "outtmpl": str(AUDIO / "%(id)s.%(ext)s"),
    "quiet": True,
    "no_warnings": True,
    "noprogress": True,
    "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}],
}

failed = []
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    for i, url in enumerate(urls, 1):
        vid = url.rstrip("/").split("/")[-1]
        if (OUT / f"{vid}.json").exists():
            continue
        try:
            info = ydl.extract_info(url, download=True)
            segments, _ = model.transcribe(load_audio(AUDIO / f"{info['id']}.mp3"))
            transcript = " ".join(s.text.strip() for s in segments)
            (OUT / f"{vid}.json").write_text(json.dumps({
                "id": info["id"],
                "url": url,
                "uploader": info.get("uploader"),
                "title": info.get("title"),
                "description": info.get("description"),
                "view_count": info.get("view_count"),
                "like_count": info.get("like_count"),
                "upload_date": info.get("upload_date"),
                "transcript": transcript,
            }, ensure_ascii=False, indent=2))
            print(f"[{i}/{len(urls)}] ok {vid}", flush=True)
        except Exception as e:
            failed.append(url)
            print(f"[{i}/{len(urls)}] FAIL {vid}: {e}", file=sys.stderr, flush=True)
        time.sleep(2)  # be gentle with TikTok

print(f"done; {len(failed)} failed")
