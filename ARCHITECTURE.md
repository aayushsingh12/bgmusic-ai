# bgmusic-ai — Full Pipeline Architecture

> An AI-powered multi-agent system that watches any video, understands its emotional arc, and automatically composes + layers a cinematic background music soundtrack on top of it.

---

## System Overview

```
Video File (.mp4)
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│                     LangGraph StateGraph                         │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │  Transcriber │───►│ Scene Analyst│───►│ Emotion Expert   │  │
│  └──────────────┘    └──────────────┘    └──────────────────┘  │
│   (Whisper STT)       (Gemini Vision)     (Gemini + LCEL)  │    │
│                                                   │              │
│                                          ┌────────▼──────┐      │
│                                          │   Visualizer  │      │
│                                          └────────┬──────┘      │
│                                                   │              │
│                                          ┌────────▼──────┐      │
│                                          │   Composer    │      │
│                                          └────────┬──────┘      │
│                                                   │              │
│                                          ┌────────▼──────┐      │
│                                          │ Video Muxer   │      │
│                                          └───────────────┘      │
└─────────────────────────────────────────────────────────────────┘
       │
       ▼
final_output.mp4  +  scene_timeline.png  +  full_scene_graph.md
```

---

## Orchestration Layer: LangGraph

**File:** `src/graph.py`

The entire pipeline runs inside a **LangGraph `StateGraph`** — a directed acyclic graph where each node is an AI agent. A single shared **`AgentState`** TypedDict object flows through the graph, getting enriched by each node.

### AgentState (Shared Memory)

```python
class AgentState(TypedDict):
    video_path: str                          # Input: path to the video file
    transcript: Dict[str, Any]               # Output of Transcriber (Whisper)
    scene_graph: Dict[str, Any]              # Output of SceneAnalyst
    emotions_with_timestamps: List[Dict]     # Output of EmotionExpert
    music_path: str                          # Output of Composer
    visualizer_path: str                     # Output of Visualizer
    final_video_path: str                    # Output of VideoMuxer
    error: str                               # Any failure message
```

### Error Routing

Every edge in the graph is a **conditional edge**. After each node finishes, the router checks if `state["error"]` is set. If there's an error, the graph immediately terminates and reports the first failure. This prevents silent cascading failures downstream.

```
Node A ──► route_fn ──► [error] ──► END (reports error)
                    ──► [ok]    ──► Node B
```

---

## The Agents

### 0. 🎙️ Transcriber  *(NEW — runs first)*
**File:** `src/agents/transcriber.py`  
**Model:** `openai/whisper-base` via Hugging Face `openai-whisper`

**What it does:**  
Extracts the audio track from the video using `moviepy`, writes it to a temporary `.wav` file, then runs OpenAI Whisper to perform speech-to-text transcription. Whisper auto-detects the spoken language and returns both a full transcript and per-segment timestamps.

**Input:** `video_path`  
**Output:** `transcript` (dict with `full_text`, `segments`, `language`)

```
Video File
    │
    ▼ moviepy VideoFileClip → .audio.write_audiofile(tmp.wav)
Temp audio file (WAV)
    │
    ▼ whisper.load_model("base")
    ▼ model.transcribe(tmp.wav)
transcript = {
  "language": "hi",               # auto-detected
  "full_text": "Yeh rasta mushkil hai...",
  "segments": [
    {"start": 0.0, "end": 3.2, "text": "Yeh rasta mushkil hai"},
    {"start": 3.2, "end": 6.1, "text": "Par hum haar nahi mante"},
    ...
  ]
}
```

> **Note:** Transcription failures are **non-fatal**. If a video has no audio or Whisper fails, the pipeline continues using pure visual analysis.

---

### 1. 🎬 Scene Analyst
**File:** `src/agents/scene_analyst.py`  
**Model:** `gemini-2.5-flash` via `langchain-google-genai`

**What it does:**  
Opens the video and physically extracts one frame every 5 seconds using `cv2` (OpenCV). Each frame is converted to a JPEG image, base64-encoded, and packaged as a `HumanMessage` with multimodal image content for Gemini.

**Input:** `video_path` (string)  
**Output:** `scene_graph` (dict)

```
Video File
    │
    ▼ cv2.VideoCapture
Extract frame at t=0s, 5s, 10s, 15s... (JPEG)
    │
    ▼ base64 encode each frame
Pack as LangChain HumanMessage (text prompt + image list)
    │
    ▼ ChatGoogleGenerativeAI (gemini-2.5-flash)
"Describe each frame: actions, characters, setting..."
    │
    ▼ Split response on newlines
scene_graph = {
  "events": [
    {"timestamp": 0,  "description": "A man sits at a desk, head bowed..."},
    {"timestamp": 5,  "description": "Close-up of handwritten notes..."},
    ...
  ]
}
```

> **Note:** The agent only sees **silent visual frames**. It has no access to the audio track or dialogue.

---

### 2. 🧠 Emotion Expert
**File:** `src/agents/emotion_expert.py`  
**Model:** `gemini-2.5-flash` via `langchain-google-genai`

**What it does:**  
Reads both the `scene_graph` events AND the `transcript` and asks Gemini to assign a single dominant emotion to each timestamp using **both** sources of context. Uses a proper **LangChain LCEL chain** with a `ChatPromptTemplate` and a `JsonOutputParser`.

**Input:** `scene_graph` + `transcript`  
**Output:** `emotions_with_timestamps` (list of dicts)

```
scene_graph.events  +  transcript.segments
    │
    ▼ Format both as structured text
ChatPromptTemplate:
  system: "Use BOTH visual context AND dialogue to identify emotions..."
  human:  "VISUAL: [descriptions]\nDIALOGUE: [transcript segments]"
    │
    ▼ ChatGoogleGenerativeAI (gemini-2.5-flash)
    │
    ▼ JsonOutputParser (enforces JSON schema)
[
  {"timestamp": "0s",  "emotion": "Determination"},
  {"timestamp": "5s",  "emotion": "Respect"},
  ...
]
```

**Chain (LCEL):**
```python
chain = prompt | model | JsonOutputParser()
```

---

### 3. 📊 Visualizer
**File:** `src/visualizer.py`

**What it does:**  
Generates two outputs for human inspection:

**a) `scene_timeline.png`** — A matplotlib scatter-line chart plotting each emotion label on a horizontal time axis. Purple dots at each timestamp, labeled with the identified emotion rotated 45°.

**b) `full_scene_graph.md`** — A markdown table that joins the raw visual descriptions from the Scene Analyst with the emotions from the Emotion Expert, giving you the full picture:

| Timestamp | Description | Emotion |
|-----------|-------------|---------|
| 0s | A man sits at a desk... | **Despair** |
| 5s | Close-up of handwritten notes... | **Respect** |

**Input:** `scene_graph` + `emotions_with_timestamps`  
**Output:** `visualizer_path` (path to the PNG)

---

### 4. 🎵 Composer
**File:** `src/agents/composer.py`  
**Model:** `facebook/musicgen-small` via Hugging Face `transformers`

**What it does:**  
This is the creative heart of the pipeline. It reads the full emotion timeline, **aggregates it into a single cinematic music prompt**, and runs it through Facebook's MusicGen generative audio model locally on your machine.

**Input:** `emotions_with_timestamps`  
**Output:** `music_path` (`generated_soundtrack.wav`)

```
emotions_with_timestamps
    │
    ▼ Sort by timestamp, deduplicate consecutive same emotions
["Despair", "Distress", "Grief", "Hope", "Shock"]
    │
    ▼ Take first 5 distinct transitions
Prompt = "An orchestral cinematic movie soundtrack that evokes:
          Despair then Distress then Grief then Hope. High quality."
    │
    ▼ AutoProcessor (tokenizes text prompt)
    │
    ▼ MusicgenForConditionalGeneration.generate(max_new_tokens=1000)
    │   (~20-30 seconds of audio at 32kHz)
    ▼
scipy.io.wavfile.write("generated_soundtrack.wav")
```

**Hardware:** Automatically uses GPU (`cuda`) if available, otherwise CPU. First run downloads ~2.4GB model weights from Hugging Face (cached for subsequent runs).

---

### 5. 📼 Video Muxer
**File:** `src/video_muxer.py`  
**Library:** `moviepy`

**What it does:**  
The final assembly step. Takes the original video and the AI-composed `.wav` file. Since MusicGen generates ~20-30 seconds of audio but the video may be much longer, it **loops the soundtrack** seamlessly to cover the entire video duration, then renders out the final MP4.

**Input:** `video_path` + `music_path`  
**Output:** `final_video_path` (`final_output.mp4`)

```
original_video.mp4  +  generated_soundtrack.wav
         │                        │
         │         moviepy        │
         │  ◄──────────────────►  │
         │                        │
         ▼  AudioLoop(duration=video.duration)
    looped_audio (~110s if video is 110s)
         │
         ▼  video.with_audio(looped_audio)
         │
         ▼  write_videofile(codec="libx264", audio_codec="aac")
    final_output.mp4 ✅
```

---

## Full Data Flow Diagram

```
main.py
  │  {"video_path": "clip.mp4"}
  ▼
[LANGGRAPH START]
  │
  ├──► transcriber_node           ← NEW
  │      moviepy extracts audio to temp WAV
  │      Whisper auto-detects language, transcribes speech
  │      → AgentState["transcript"] updated (non-fatal if fails)
  │
  ├──► scene_analyst_node
  │      cv2 extracts frames every 5s
  │      Gemini (gemini-2.5-flash) describes each frame visually
  │      → AgentState["scene_graph"] updated
  │
  ├──► emotion_expert_node
  │      LangChain LCEL chain runs
  │      Gemini receives BOTH visual descriptions AND dialogue transcript
  │      JsonOutputParser enforces schema
  │      → AgentState["emotions_with_timestamps"] updated
  │
  ├──► visualizer_node
  │      matplotlib draws emotion timeline → scene_timeline.png
  │      Writes joined markdown table → full_scene_graph.md
  │      → AgentState["visualizer_path"] updated
  │
  ├──► composer_node
  │      Deduplicates emotions into narrative arc
  │      Builds single cinematic prompt
  │      MusicGen synthesizes WAV locally
  │      → AgentState["music_path"] updated
  │
  ├──► video_muxer_node
  │      moviepy loops WAV to video length
  │      Renders final MP4
  │      → AgentState["final_video_path"] updated
  │
[LANGGRAPH END]
  │
  ▼
Outputs:
  ✅ final_output.mp4        (video + AI background music)
  ✅ generated_soundtrack.wav (raw AI audio)
  ✅ scene_timeline.png      (emotion arc chart)
  ✅ full_scene_graph.md     (full scene descriptions + emotions)
```

---

## Technology Stack

| Component | Library / Model |
|-----------|-----------------|
| Orchestration | `langgraph` (StateGraph) |
| LLM Framework | `langchain-core`, `langchain-google-genai` |
| Vision LLM | Google `gemini-2.5-flash` |
| Text-to-Audio | `facebook/musicgen-small` (Hugging Face) |
| ML Runtime | `torch` (CPU or CUDA) |
| Video Framing | `opencv-python` (cv2) |
| Audio Writing | `scipy.io.wavfile` |
| Video Muxing | `moviepy` |
| Visualization | `matplotlib` |
| Env Config | `python-dotenv` |

---

## File Structure

```
bgmusic-ai/
├── .env                          # GEMINI_API_KEY lives here
├── src/
│   ├── main.py                   # Entrypoint — set video_path here
│   ├── graph.py                  # LangGraph definition (nodes + edges)
│   ├── visualizer.py             # matplotlib + markdown report generator
│   ├── video_muxer.py            # moviepy audio/video combiner
│   └── agents/
│       ├── transcriber.py        # Agent 0: Video audio → Whisper transcript  ← NEW
│       ├── scene_analyst.py      # Agent 1: Video frames → Scene Graph
│       ├── emotion_expert.py     # Agent 2: Scene Graph + Transcript → Emotions
│       └── composer.py           # Agent 3: Emotions → WAV file
│
├── final_output.mp4              # ← YOUR FINAL VIDEO IS HERE
├── generated_soundtrack.wav      # Raw AI audio
├── scene_timeline.png            # Emotion arc chart
└── full_scene_graph.md           # Full AI scene interpretation table
```

---

## Running the Pipeline

```bash
# Set your video file in src/main.py:
# video_path = "your_clip.mp4"

python -m src.main
```

First run will download the MusicGen model weights (~2.4GB). Subsequent runs use the local cache. GPU recommended for faster inference, but CPU works fine.
