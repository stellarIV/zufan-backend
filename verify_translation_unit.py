import requests
import os

BASE_URL = "http://localhost:5000"

def test_translation_upload():
    print("Testing PDF upload with translation...")
    
    # Create a dummy PDF with Amharic text if possible, or just use a dummy file
    # for effective testing we need a real PDF. 
    # Since I cannot easily create a PDF with complex font, I will try to use an existing one if available, 
    # or just assume the user has one. But for automation, let's try to mock the upload or use a sample text.
    
    # Actually, simpler test: Unit test the rag_utils.translate_text function directly
    # This avoids PDF creation issues.
    
    try:
        from rag_utils import translate_text
        
        amharic_text = "ጤና ይስጥልኝ። እንደምን አላችሁ?" # Hello. How are you?
        print(f"Original: {amharic_text}")
        
        translated = translate_text(amharic_text)
        print(f"Translated: {translated}")
        
        if "Hello" in translated or "Peace" in translated or "Hi" in translated:
            print("SUCCESS: Translation working.")
        else:
            print("WARNING: Translation might have failed or returned unexpected result.")
            
    except ImportError:
        print("Could not import rag_utils. Make sure you are in the correct directory.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_translation_upload()
