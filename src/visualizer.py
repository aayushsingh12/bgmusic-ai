import matplotlib.pyplot as plt
import os

class Visualizer:
    def __init__(self):
        pass

    def generate_timeline(self, scene_graph, emotions_with_timestamps, output_path="scene_timeline.png"):
        """
        Generates a timeline plot of scenes and emotions.
        """
        if not scene_graph or "events" not in scene_graph:
            return {"error": "Missing scene graph data for visualization."}
        if not emotions_with_timestamps:
            return {"error": "Missing emotion data for visualization."}

        timestamps = []
        emotions = []
        
        for item in emotions_with_timestamps:
            ts_val = item.get("timestamp", 0)
            if isinstance(ts_val, str):
                ts_val = float(ts_val.replace('s', ''))
            timestamps.append(ts_val)
            emotions.append(item.get("emotion", "Unknown"))
            
        if not timestamps:
            return {"error": "Could not parse timestamps for visualization."}

        # Setup figure
        plt.figure(figsize=(14, 6))
        
        # We create a simple scatter plot along a horizontal line
        y_values = [1] * len(timestamps)
        plt.scatter(timestamps, y_values, color='#6c35dc', s=100, zorder=3)
        plt.plot(timestamps, y_values, color='#8b949e', linestyle='-', zorder=2)

        # Annotate each point with the emotion
        for i, txt in enumerate(emotions):
            plt.annotate(
                txt, 
                (timestamps[i], y_values[i]), 
                textcoords="offset points", 
                xytext=(0, 15), 
                ha='center', 
                rotation=45,
                fontsize=10,
                fontweight='bold',
                color='#2d333b'
            )
            
        plt.xlabel('Time (seconds)', fontsize=12)
        plt.yticks([]) # Hide Y axis ticks since it's a 1D timeline
        plt.title('Determined Emotions Timeline', fontsize=16)
        plt.grid(True, axis='x', linestyle='--', alpha=0.6)
        
        # Ensure nothing gets clipped
        plt.tight_layout()
        
        # Save visualization to disk
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        print(f"Visualization generated successfully at: {output_path}")
        return output_path

    def save_transcript_text(self, transcript, output_path="transcript.txt"):
        """
        Saves the full transcript text to a file.
        """
        if not transcript or "full_text" not in transcript:
            return
            
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(transcript["full_text"])
        print(f"Transcript saved to: {output_path}")

    def save_emotions_json(self, emotions, output_path="emotions.json"):
        """
        Saves the raw emotions list to a JSON file.
        """
        import json
        if not emotions:
            return
            
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(emotions, f, indent=4)
        print(f"Emotions JSON saved to: {output_path}")

    def generate_markdown_report(self, scene_graph, emotions_with_timestamps, output_path="scene_graph.md"):

        """
        Generates a markdown table mapping the full scene graph descriptions to the emotions.
        """
        if not scene_graph or "events" not in scene_graph:
            return {"error": "Missing scene graph data for md report."}
        
        # Emulate a join on timestamp
        emotion_map = {}
        if emotions_with_timestamps:
            for item in emotions_with_timestamps:
                ts_val = item.get("timestamp", 0)
                if isinstance(ts_val, str):
                    ts_val = ts_val.replace('s', '')
                emotion_map[str(ts_val)] = item.get("emotion", "Unknown")

        md_content = "# Full Generated Scene Graph\n\n"
        md_content += f"**Description**: {scene_graph.get('description', '')}\n\n"
        md_content += "| Timestamp | Description | Emotion |\n"
        md_content += "| --- | --- | --- |\n"
        
        for event in scene_graph["events"]:
            ts_str = str(event.get('timestamp', 0))
            desc = event.get('description', '').replace('\\n', ' ')
            emo = emotion_map.get(ts_str, "N/A")
            md_content += f"| {ts_str}s | {desc} | **{emo}** |\n"

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
            
        print(f"Markdown report generated successfully at: {output_path}")
        return output_path

