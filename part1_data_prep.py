import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def prepare_data(pdf_path):
    """
    Part 1 deliverable:
    Load a PDF, clean its text, and split it into overlapping chunks.
    """
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"Could not find PDF: {pdf_file}")

    print(f"Loading document: {pdf_file}...")

    loader = PyPDFLoader(str(pdf_file))
    pages = loader.load()

    for page in pages:
        page.page_content = clean_text(page.page_content)

    print(f"Successfully loaded {len(pages)} pages.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    chunks = text_splitter.split_documents(pages)
    print(f"Successfully split the document into {len(chunks)} chunks.")

    if chunks:
        print("\n--- Preview of Chunk 1 ---")
        print(chunks[0].page_content[:1000])
        print("--------------------------\n")

    return chunks


def clean_text(text):
    """Basic NLP cleaning: remove repeated spaces, tabs, and line breaks."""
    return re.sub(r"\s+", " ", text).strip()


if __name__ == "__main__":
    sample_pdf_path = "sample.pdf"

    try:
        prepare_data(sample_pdf_path)
    except FileNotFoundError as error:
        print(f"Error: {error}")
        print("Put a PDF in this folder and name it sample.pdf, or change sample_pdf_path.")
