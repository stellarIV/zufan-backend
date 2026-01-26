import os
from dotenv import load_dotenv
load_dotenv()
key = os.environ.get("GOOGLE_API_KEY")
print(f"DEBUG_KEY_FOUND: {key is not None}")
if key:
    print(f"DEBUG_KEY_LEN: {len(key)}")
    print(f"DEBUG_KEY_START: {key[:5]}...")
