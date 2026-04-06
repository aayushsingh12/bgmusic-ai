from dotenv import load_dotenv
import os
from src.graph import app

def main():
    # Load environment variables from .env file
    load_dotenv()

    # Check if the Gemini API key is set
    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable not set.")
        print("Please create a .env file and add your Gemini API key.")
        return

    # Define the video to be processed
    video_path = "Peter_Parker_Evil_s_Dance_Scene_-_Spider-Man_3_2007_Movie_CLIP_HD_1080p.mp4" 

    # Define the initial state
    initial_state = {"video_path": video_path}

    # Run the graph
    final_state = app.invoke(initial_state)

    # Print the final result
    if "error" in final_state and final_state["error"]:
        print(f"\nAn error occurred: {final_state['error']}")
    else:
        print("\n--- Final Results (Terminal Mode) ---")
        
        print("\n[🎙️ TRANSCRIPT]")
        transcript = final_state.get("transcript", {})
        if transcript and "full_text" in transcript:
            print(f"Language: {transcript.get('language')}")
            print(f"Text: {transcript.get('full_text')}")
        else:
            print("No transcript detected.")

        print("\n[🔍 SCENE GRAPH]")
        scene_graph = final_state.get("scene_graph", {})
        if scene_graph and "events" in scene_graph:
            for ev in scene_graph["events"]:
                print(f"Time: {ev['timestamp']}s | {ev['description']}")
        
        print("\n[🧠 EMOTIONS]")
        print(final_state.get("emotions_with_timestamps"))
        
        print("\n[🎵 OUTPUT FILES]")
        print(f"Generated Music: {final_state.get('music_path')}")
        print(f"Final Video:     {final_state.get('final_video_path')}")
        print(f"Timeline Chart:  {final_state.get('visualizer_path')}")


if __name__ == "__main__":
    main()
