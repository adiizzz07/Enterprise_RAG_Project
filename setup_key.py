from pathlib import Path


env_path = Path(".env")

print("Paste your Gemini API key below.")
print("It usually starts with AIza. The key will be saved only in this local .env file.")
api_key = input("GOOGLE_API_KEY: ").strip()

if not api_key:
    raise SystemExit("No key entered. .env was not changed.")

env_path.write_text(f"GOOGLE_API_KEY={api_key}\n", encoding="utf-8")
print(".env created successfully.")
print("Now run: python gemini_test.py")
