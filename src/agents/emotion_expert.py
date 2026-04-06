from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os
import json

class EmotionExpert:
    def __init__(self):
        self.model = ChatGoogleGenerativeAI(model='gemini-2.5-flash', api_key=os.environ.get("GEMINI_API_KEY"))
        self.parser = JsonOutputParser()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "You are an expert film emotion analyst. Analyze a sequence of visual scene descriptions "
             "from a video, combined with the spoken dialogue/transcript from those same moments. "
             "Use BOTH the visual context AND what the characters are actually saying to identify the "
             "primary emotion for each timestamp. "
             "CRITICAL: Return the result AS ONLY A JSON LIST of objects. Each object MUST have 'timestamp' and 'emotion' keys. "
             "Do not include any preamble, markdown formatting outside the JSON, or post-analysis text.\n"
             "{format_instructions}"
            ),
            ("human", 
             "VISUAL SCENE DESCRIPTIONS:\n{events_text}\n\n"
             "SPOKEN DIALOGUE / TRANSCRIPT:\n{transcript_text}\n\n"
             "Identify the emotion at each timestamp provided in the VISUAL section. "
             "Respond with ONLY the JSON list."
            )
        ])
        self.chain = self.prompt | self.model | self.parser

    def analyze_scene_graph(self, scene_graph, transcript=None):
        """
        Analyzes the scene graph and transcript, returns a list of emotions with timestamps.
        """
        
        if not scene_graph or "events" not in scene_graph or not scene_graph["events"]:
            return {"error": "Invalid or empty scene graph provided."}

        events_text = ""
        for event in scene_graph["events"]:
            events_text += f"Timestamp: {event['timestamp']}s, Description: {event['description']}\n"

        # Build transcript context string
        transcript_text = "(No transcript available — visual analysis only)"
        if transcript and "segments" in transcript and transcript["segments"]:
            transcript_text = ""
            for seg in transcript["segments"]:
                transcript_text += f"[{seg['start']}s - {seg['end']}s]: {seg['text']}\n"
        elif transcript and "full_text" in transcript and transcript["full_text"]:
            transcript_text = transcript["full_text"]
        
        try:
            emotions_with_timestamps = self.chain.invoke({
                "events_text": events_text,
                "transcript_text": transcript_text,
                "format_instructions": self.parser.get_format_instructions()
            })
            print(f"Parsed LangChain response: {emotions_with_timestamps}") # DEBUG
            
            return emotions_with_timestamps

        except Exception as e:
            print(f"Error analyzing scene graph for emotions: {e}")
            return {"error": str(e)}
