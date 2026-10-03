import os
from dotenv import load_dotenv
import openai


load_dotenv() 


try:
    
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")) 
    
    print("Connecting to OpenAI and requesting a simple mood prompt...")
    
    
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct", 
        prompt="Write a single, descriptive word for a calm mood.",
        max_tokens=5 # Keep cost low
    )
    
    print("\n---  API Connection Successful! ---")
    print(f"Generated Word: {response.choices[0].text.strip()}")

except Exception as e:
    print(f"\n---  ERROR: Connection Failed! ---")
    print("Please check your OPENAI_API_KEY in the .env file.")
    print(f"Details: {e}")