# Enterprise RAG Project

This project builds a simple Retrieval-Augmented Generation (RAG) system for a private PDF document.

The project follows the two assignment deliverables:

1. Part 1: Load a raw PDF, clean the text, and split it into overlapping chunks.
2. Part 2: Convert the chunks into a vector database, retrieve the top 3 relevant chunks, and use Gemini to answer questions using only the PDF context.

## Project Files

- `part1_data_prep.py` - Part 1 data preparation script.
- `app.py` - Part 2 end-to-end RAG pipeline.
- `gemini_test.py` - Small script to test if Gemini is connected.
- `setup_key.py` - Helper script to save your Gemini key in `.env`.
- `requirements.txt` - Required Python libraries.
- `sample.pdf` - The private document used by the RAG system.

## Setup

Install the required libraries:

```powershell
python -m pip install -r requirements.txt
```

Save your Gemini API key:

```powershell
python setup_key.py
```

Test Gemini:

```powershell
python gemini_test.py
```

## Run Part 1

```powershell
python part1_data_prep.py
```

This script:

- Loads `sample.pdf`
- Cleans extra spaces and line breaks
- Splits the document into chunks
- Prints a preview of the first chunk

## Run Part 2

```powershell
python app.py
```

This script:

- Runs the Part 1 data preparation
- Creates embeddings with `all-MiniLM-L6-v2`
- Stores the embeddings in Chroma
- Retrieves the top 3 relevant chunks
- Sends those chunks to Gemini
- Prints the answer and retrieved source chunks

## Example Questions

```text
what is the cheapest AM method
```

```text
what is the angle when the z-component of the unit normal is less than -0.5
```

```text
what are the objectives of this project
```

## Important Notes

- The answer is generated only from the retrieved PDF chunks.
- If the answer is not in the PDF, the model should say it does not know.
- The `.env` file contains your private API key and should not be shared.
