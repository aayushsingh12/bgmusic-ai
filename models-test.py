from google import genai
from dotenv import load_dotenv
import os

load_dotenv()
client = genai.Client()

print("--- Available Models for your API Key ---")

# In the new SDK, we use model.supported_actions
for model in client.models.list():
    if 'generateContent' in model.supported_actions:
        # Use model.name (e.g., 'models/gemini-2.0-flash')
        print(f"Model ID: {model.name}")
