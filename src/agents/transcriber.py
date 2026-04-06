import whisper
import os
import tempfile
import subprocess
import imageio_ffmpeg
import shutil
import pathlib

# Inject bundled ffmpeg binary into PATH at import time so Whisper can find it
# On Windows, Whisper specifically looks for 'ffmpeg' or 'ffmpeg.exe'.
# The bundled imageio_ffmpeg binary has a different name, so we'll create a temp copy.
_ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
_ffmpeg_dir = os.path.dirname(_ffmpeg_exe)
_tmp_bin_dir = os.path.join(tempfile.gettempdir(), "bgmusic_ai_bin")
os.makedirs(_tmp_bin_dir, exist_ok=True)
_dest_ffmpeg = os.path.join(_tmp_bin_dir, "ffmpeg.exe")

if not os.path.exists(_dest_ffmpeg):
    try:
        shutil.copy2(_ffmpeg_exe, _dest_ffmpeg)
    except Exception as e:
        print(f"Warning: Could not create ffmpeg alias: {e}")

if _tmp_bin_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _tmp_bin_dir + os.pathsep + os.environ.get("PATH", "")


class Transcriber:
    def __init__(self, model_size="base"):
        """
        Loads the Whisper model. model_size can be:
        'tiny', 'base', 'small', 'medium', 'large'
        'base' is a good balance of speed and accuracy.
        """
        print(f"Loading Whisper model ({model_size})...")
        self.model = whisper.load_model(model_size)

    def transcribe_video(self, video_path):
        """
        Extracts audio from the video and runs Whisper speech-to-text.
        Returns a dict with full transcript text and per-segment timestamps.
        """
        if not os.path.exists(video_path):
            return {"error": f"Video file not found: {video_path}"}

        print(f"Extracting audio from {video_path}...")

        try:
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
                tmp_audio_path = tmp_audio.name
            # On Windows, the file must be closed before another process can write to it

            subprocess.run(
                [ffmpeg_exe, "-y", "-i", video_path, "-ar", "16000", "-ac", "1", "-f", "wav", tmp_audio_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )

            print("Running Whisper transcription...")
            result = self.model.transcribe(tmp_audio_path, fp16=False)

            # Clean up temp file
            os.unlink(tmp_audio_path)

            # Structure the output
            transcript_segments = []
            for seg in result["segments"]:
                transcript_segments.append({
                    "start": round(seg["start"], 1),
                    "end": round(seg["end"], 1),
                    "text": seg["text"].strip()
                })

            transcript_data = {
                "full_text": result["text"].strip(),
                "segments": transcript_segments,
                "language": result.get("language", "unknown")
            }

            print(f"Transcription complete. Language detected: {transcript_data['language']}")
            print(f"Transcript preview: {transcript_data['full_text'][:200]}...")

            return transcript_data

        except Exception as e:
            # Clean up temp file if something went wrong
            if 'tmp_audio_path' in locals() and os.path.exists(tmp_audio_path):
                os.unlink(tmp_audio_path)
            return {"error": f"Transcription failed: {e}"}
