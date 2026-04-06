import os
import torch
import scipy.io.wavfile
from transformers import AutoProcessor, MusicgenForConditionalGeneration

class Composer:
    def __init__(self):
        print("Loading MusicGen model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        self.model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").to(self.device)

    def compose_music(self, emotions_with_timestamps):
        """
        Composes music based on a timeline of emotions.
        """
        if not emotions_with_timestamps:
            return {"error": "No emotions provided to compose music."}

        print("Starting music composition process...")
        
        ordered_emotions = sorted(emotions_with_timestamps, key=lambda x: float(str(x['timestamp']).replace('s', '')))
        
        # Format the emotions timeline into a readable string
        emotion_timeline_str = ", ".join([f"{str(item['timestamp']).replace('s', '')}s: {item['emotion']}" for item in ordered_emotions])
        
        print("Synthesizing music prompt via Gemini...")
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_core.messages import HumanMessage
            llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash', api_key=os.environ.get("GEMINI_API_KEY"))
            sys_prompt = "You are an expert music director. Synthesize the provided emotion timeline into a cohesive 1-2 sentence prompt for an AI music generator (under 200 characters). Describe the musical style, genre, and how the mood evolves from the beginning, middle, to end of the scene. Include 'High quality.' at the end."
            message = HumanMessage(content=f"{sys_prompt}\n\nTimeline:\n{emotion_timeline_str}")
            response = llm.invoke([message])
            prompt = response.content.strip()
            # Remove any wrapping quotes if present
            if prompt.startswith('"') and prompt.endswith('"'):
                prompt = prompt[1:-1]
        except Exception as e:
            print(f"Failed to generate dynamic prompt ({e}), falling back to default.")
            summary_path = []
            last_emo = None
            for item in ordered_emotions:
                if item['emotion'] != last_emo:
                    summary_path.append(item['emotion'])
                    last_emo = item['emotion']
            prompt = f"An orchestral cinematic movie soundtrack that evokes: {' then '.join(summary_path[:5])}. High quality."
            
        print(f"Aggregated Prompt: {prompt}")
        
        # Save the prompt to a text file
        with open("musicgen_prompt.txt", "w", encoding="utf-8") as f:
            f.write(prompt)

        try:
            inputs = self.processor(
                text=[prompt],
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            print("Generating audio via MusicGen (this may take a while)...")
            audio_values = self.model.generate(**inputs, max_new_tokens=1000)
            
            output_path = "generated_soundtrack.wav"
            sampling_rate = self.model.config.audio_encoder.sampling_rate
            
            audio_data = audio_values[0, 0].cpu().numpy()
            scipy.io.wavfile.write(output_path, rate=sampling_rate, data=audio_data)
            
            print(f"\nMusic composition complete. Final soundtrack at: {output_path}")
            return output_path
            
        except Exception as e:
            return {"error": f"Failed to generate music: {e}"}
