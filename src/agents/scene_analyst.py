import cv2
import base64
import os
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from PIL import Image
import io

class SceneAnalyst:
    def __init__(self):
        self.model = ChatGoogleGenerativeAI(model='gemini-2.5-flash', api_key=os.environ.get("GEMINI_API_KEY"))

    def analyze_video(self, video_path, interval=1):
        """
        Analyzes the video file, extracts frames, and returns a scene graph.
        """
        if not os.path.exists(video_path):
            return {"error": "Video file not found."}

        video = cv2.VideoCapture(video_path)
        frames = []
        timestamps = []

        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps

        seconds = 0
        while video.isOpened() and seconds < duration:
            video.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
            success, frame = video.read()
            if not success:
                break
            
            # Convert frame to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            frames.append(pil_img)
            timestamps.append(seconds)
            seconds += interval
        
        video.release()

        if not frames:
            return {"error": "Could not extract frames from the video."}

        # Generate descriptions for the frames
        descriptions_result = self._generate_descriptions(frames)
        if isinstance(descriptions_result, dict) and "error" in descriptions_result:
            return descriptions_result
        
        # Construct the scene graph
        scene_graph = self._create_scene_graph(descriptions_result, timestamps)

        return scene_graph

    def _generate_descriptions(self, frames):
        """
        Uses Gemini to generate descriptions for each frame.
        """
        prompt = "These are frames from a video. Generate a short description for each frame, focusing on actions, characters, and the setting. Provide each description on a new line."
        
        content = [{"type": "text", "text": prompt}]
        for frame in frames:
            buffered = io.BytesIO()
            frame.save(buffered, format="JPEG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{img_b64}"
            })
        
        try:
            message = HumanMessage(content=content)
            response = self.model.invoke([message])
            return response.content.split('\n')
        except Exception as e:
            print(f"Error generating descriptions: {e}")
            return {"error": str(e)}


    def _create_scene_graph(self, descriptions, timestamps):
        """
        Creates a scene graph from the descriptions and timestamps.
        """
        scene_graph = {
            "description": "A scene graph generated from video analysis.",
            "events": []
        }

        for i, desc in enumerate(descriptions):
            if i < len(timestamps):
                scene_graph["events"].append({
                    "timestamp": timestamps[i],
                    "description": desc
                })
        
        return scene_graph
