from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from src.agents.scene_analyst import SceneAnalyst
from src.agents.emotion_expert import EmotionExpert
from src.agents.composer import Composer
from src.agents.transcriber import Transcriber
from src.visualizer import Visualizer
from src.video_muxer import VideoMuxer

# Define the state for our graph
class AgentState(TypedDict):
    video_path: str
    transcript: Dict[str, Any]
    scene_graph: Dict[str, Any]
    emotions_with_timestamps: List[Dict[str, Any]]
    music_path: str
    visualizer_path: str
    final_video_path: str
    error: str

# Define the agent nodes
def transcriber_node(state: AgentState):
    """
    Extracts and transcribes the audio from the video using Whisper.
    """
    print("--- Node: Transcriber ---")
    video_path = state.get("video_path")
    if not video_path:
        return {"error": "Video path not provided."}

    transcriber = Transcriber(model_size="base")
    transcript = transcriber.transcribe_video(video_path)

    if "error" in transcript:
        # Transcription failure is non-fatal — we continue with visual analysis only
        print(f"Warning: Transcription failed ({transcript['error']}). Continuing with visual-only analysis.")
        return {"transcript": {}}

    return {"transcript": transcript}

def scene_analysis_node(state: AgentState):
    """
    Analyzes the video to create a scene graph.
    """
    print("--- Node: Scene Analyst ---")
    video_path = state.get("video_path")
    if not video_path:
        return {"error": "Video path not provided."}

    analyst = SceneAnalyst()
    scene_graph = analyst.analyze_video(video_path, interval=5)
    
    if "error" in scene_graph:
        return {"error": f"Scene Analysis failed: {scene_graph['error']}"}

    return {"scene_graph": scene_graph}

def emotion_expert_node(state: AgentState):
    """
    Identifies emotions from the scene graph.
    """
    print("--- Node: Emotion Expert ---")
    scene_graph = state.get("scene_graph")
    if not scene_graph:
        return {"error": "Scene graph not provided."}

    expert = EmotionExpert()
    transcript = state.get("transcript")  # May be empty dict if transcription failed
    emotions = expert.analyze_scene_graph(scene_graph, transcript=transcript)

    if "error" in emotions:
        return {"error": f"Emotion Interpretation failed: {emotions['error']}"}
    
    return {"emotions_with_timestamps": emotions}

def composer_node(state: AgentState):
    """
    Composes music based on the identified emotions.
    """
    print("--- Node: Composer ---")
    emotions = state.get("emotions_with_timestamps")
    if not emotions:
        return {"error": "Emotions not provided."}

    composer = Composer()
    music_path = composer.compose_music(emotions)

    if "error" in music_path:
        return {"error": f"Music Composition failed: {music_path['error']}"}

    return {"music_path": music_path}

def visualizer_node(state: AgentState):
    """
    Generates a visual timeline of the scene graph.
    """
    print("--- Node: Visualizer ---")
    scene_graph = state.get("scene_graph")
    emotions = state.get("emotions_with_timestamps")
    
    if not emotions or not scene_graph:
        return {"error": "Missing data for visualizer."}

    vis = Visualizer()
    result = vis.generate_timeline(scene_graph, emotions)
    vis.generate_markdown_report(scene_graph, emotions)
    
    # Save separate files as requested
    transcript = state.get("transcript")
    if transcript:
        vis.save_transcript_text(transcript)
    vis.save_emotions_json(emotions)
    
    if isinstance(result, dict) and "error" in result:
        return {"error": f"Visualization failed: {result['error']}"}

    return {"visualizer_path": result}

def video_muxer_node(state: AgentState):
    """
    Combines the generated audio with the original video.
    """
    print("--- Node: Video Muxer ---")
    video_path = state.get("video_path")
    music_path = state.get("music_path")

    if not video_path or not music_path:
        return {"error": "Missing video or music path for muxing."}

    muxer = VideoMuxer()
    result = muxer.mux_audio_video(video_path, music_path)

    if isinstance(result, dict) and "error" in result:
        return {"error": f"Video Muxing failed: {result['error']}"}

    return {"final_video_path": result}


# Define the graph
workflow = StateGraph(AgentState)

# Add the nodes
workflow.add_node("transcriber", transcriber_node)
workflow.add_node("scene_analyst", scene_analysis_node)
workflow.add_node("emotion_expert", emotion_expert_node)
workflow.add_node("composer", composer_node)
workflow.add_node("visualizer", visualizer_node)
workflow.add_node("video_muxer", video_muxer_node)

# Set the entrypoint — transcriber runs first
workflow.set_entry_point("transcriber")

# Add the edges
def route_after_transcriber(state: AgentState):
    # Transcription errors are non-fatal, always continue to scene analysis
    if state.get("error"):
        return "end"
    return "continue"

def route_after_analyst(state: AgentState):
    if state.get("error"):
        return "end"
    return "continue"

def route_after_expert(state: AgentState):
    if state.get("error"):
        return "end"
    return "continue"

def route_after_visualizer(state: AgentState):
    if state.get("error"):
        return "end"
    return "continue"

def route_after_composer(state: AgentState):
    if state.get("error"):
        return "end"
    return "continue"

workflow.add_conditional_edges(
    "transcriber",
    route_after_transcriber,
    {"continue": "scene_analyst", "end": END}
)

workflow.add_conditional_edges(
    "scene_analyst",
    route_after_analyst,
    {"continue": "emotion_expert", "end": END}
)

workflow.add_conditional_edges(
    "emotion_expert",
    route_after_expert,
    {"continue": "visualizer", "end": END}
)

workflow.add_conditional_edges(
    "visualizer",
    route_after_visualizer,
    {"continue": "composer", "end": END}
)

workflow.add_conditional_edges(
    "composer",
    route_after_composer,
    {"continue": "video_muxer", "end": END}
)

workflow.add_edge("video_muxer", END)

# Compile the graph
app = workflow.compile()
