import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("GOOGLE_API_KEY")

if not api_key:
    print("Error: GOOGLE_API_KEY not found in .env")
else:
    genai.configure(api_key=api_key)
    print("Writing all reachable models to models.txt...")
    try:
        with open("models.txt", "w", encoding="utf-8") as f:
            for m in genai.list_models():
                f.write(f"- Name: {m.name}\n")
                f.write(f"  Display: {m.display_name}\n")
                f.write(f"  Methods: {m.supported_generation_methods}\n")
        print("Done.")
    except Exception as e:
        print(f"Error listing models: {e}")
