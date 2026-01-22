import cv2
import numpy as np
import time
import os
from mss import mss
from dotenv import load_dotenv
from google import genai
from google.genai import types

# 1. Load the .env file
load_dotenv()

# 2. Initialize Client
client = genai.Client() 

def get_vibe_from_frame(image_bytes):
    """Sends a screen capture frame to Gemini for analysis."""
    prompt = """
    Act as a Film Director. Analyze this screen capture for a music composer.
    Provide a concise brief including:
    - ATMOSPHERE: (e.g., High-Octane, Tense, Chill)
    - COLOR PALETTE: (e.g., Cyberpunk Neon, Natural Daylight)
    - KEY ACTION: (e.g., Character running, car driving, UI interaction)
    - RECOMMENDED BPM & INSTRUMENTATION: (e.g., 128 BPM, synth bass, electric guitar)
    Return ONLY the brief.
    """
    
    try:
        # Using gemini-2.5-flash which may have different quota limits
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg'),
                prompt
            ]
        )
        return response.text
    except Exception as e:
        # Handle quota errors more gracefully
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            return "QUOTA_EXCEEDED: API quota limit reached. Try again later or upgrade your plan."
        return f"ERROR: {e}"

def main():
    # 'with' statement ensures the screen capture resources are released properly
    with mss() as sct:
        # monitor[1] is the primary screen. monitor[0] is all monitors.
        monitor = sct.monitors[1]
        
        print("--- Screen Watcher is Active (Press 'q' in preview to quit) ---")

        try:
            while True:
                # 1. Grab raw screen pixels
                sct_img = sct.grab(monitor)
                
                # 2. Convert raw pixels to a NumPy array (MSS uses BGRA format)
                frame = np.array(sct_img)
                
                # 3. Convert BGRA to BGR for OpenCV compatibility
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                # Show a preview window of what the AI is "seeing"
                cv2.imshow('Screen Watcher Preview', frame)

                # 4. Resize to reduce API latency and token costs
                small_frame = cv2.resize(frame, (640, 360))
                _, buffer = cv2.imencode('.jpg', small_frame)
                
                # 5. Analyze the vibe
                vibe_brief = get_vibe_from_frame(buffer.tobytes())
                
                if "ERROR" in vibe_brief or "QUOTA_EXCEEDED" in vibe_brief:
                    if "QUOTA_EXCEEDED" in vibe_brief:
                        print(f"\n[⚠️  QUOTA LIMIT] API quota exceeded. Waiting 10 seconds before next attempt...")
                        print("Consider upgrading your API plan for higher limits.")
                    else:
                        print(f"\n[!] Sync Issue: {vibe_brief}")
                else:
                    print(f"\n[NEW SCREEN VIBE]\n{vibe_brief}")

                # Wait 10 seconds before the next capture to avoid quota limits
                if cv2.waitKey(10000) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("\nStopping Watcher...")
        finally:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
