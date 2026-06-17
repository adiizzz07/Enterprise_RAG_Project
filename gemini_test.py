import os

from langchain_google_genai import ChatGoogleGenerativeAI

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError(
        "GOOGLE_API_KEY is not set. In PowerShell, run: "
        '$env:GOOGLE_API_KEY="your_api_key_here"'
    )

print("Connecting to Gemini...")
model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
llm = ChatGoogleGenerativeAI(
    model=model_name,
    google_api_key=api_key,
    temperature=0,
)
response = llm.invoke("Say the word 'Hello'")
print("Model:", model_name)
print("Response:", response.content)
