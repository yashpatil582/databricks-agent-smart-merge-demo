"""
Quick script to check available Gemini models for a given API key.
Usage: python3 check_models.py [API_KEY]
Or set GEMINI_API_KEY environment variable
"""
import google.generativeai as genai
import sys
import os

# Get API key from command line argument or environment variable
if len(sys.argv) > 1:
    API_KEY = sys.argv[1]
elif os.getenv("GEMINI_API_KEY"):
    API_KEY = os.getenv("GEMINI_API_KEY")
else:
    print("Error: Please provide API key as argument or set GEMINI_API_KEY environment variable")
    print("Usage: python3 check_models.py YOUR_API_KEY")
    sys.exit(1)

def list_available_models():
    """List all available models for the API key."""
    try:
        genai.configure(api_key=API_KEY)
        
        print("Fetching available models...")
        print("-" * 60)
        
        # List all models
        models = genai.list_models()
        
        available_models = []
        for model in models:
            # Only show models that support generateContent
            if 'generateContent' in model.supported_generation_methods:
                model_name = model.name.replace('models/', '')
                available_models.append(model_name)
                print(f"✅ {model_name}")
                print(f"   Display name: {model.display_name}")
                print(f"   Description: {model.description}")
                print()
        
        print("-" * 60)
        print(f"\nFound {len(available_models)} available model(s):")
        for model in available_models:
            print(f"  - {model}")
        
        return available_models
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return []

if __name__ == "__main__":
    list_available_models()

