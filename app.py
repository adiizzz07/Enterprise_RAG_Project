import os
import re
import warnings

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

from part1_data_prep import prepare_data


PDF_PATH = "sample.pdf"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
TOP_K = 3


def build_vector_database(chunks):
    """
    Convert text chunks into numerical vectors and store them in Chroma.

    This is the Vector Database part of RAG.
    """
    embedding_model = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

    vector_database = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name="enterprise_rag_project",
    )

    return vector_database


def retrieve_top_chunks(vector_database, chunks, question, top_k=TOP_K):
    """
    Retrieve the top relevant chunks for the user's question.

    Semantic search is the main RAG search method.
    Keyword search is added as a small backup because PDF text can contain
    unusual symbols, tables, or spacing.
    """
    semantic_results = vector_database.similarity_search(question, k=top_k)
    keyword_results = keyword_search(chunks, question, k=top_k)

    combined_results = remove_duplicate_chunks(keyword_results + semantic_results)
    return combined_results[:top_k]


def keyword_search(chunks, question, k=TOP_K):
    """Simple keyword search backup for exact lines and PDF table text."""
    normalized_question = normalize_text(question)
    question_terms = expand_question_terms(normalized_question)

    scored_chunks = []

    for chunk in chunks:
        normalized_chunk = normalize_text(chunk.page_content)
        score = 0

        if normalized_question and normalized_question in normalized_chunk:
            score += 100

        for term, weight in question_terms.items():
            if term in normalized_chunk:
                score += weight

        if score > 0:
            scored_chunks.append((score, chunk))

    scored_chunks.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in scored_chunks[:k]]


def expand_question_terms(normalized_question):
    """
    Add simple synonyms so beginner-style questions match PDF wording.

    Example: user says "AM method", but the PDF table says "AM Process".
    """
    terms = {
        term: 1
        for term in re.findall(r"[a-zA-Z0-9]+", normalized_question)
        if len(term) > 2
    }

    if "cheapest" in normalized_question:
        terms["cheapest"] = 25
        terms["cost score"] = 15
        terms["10 (cheapest)"] = 30

    if "method" in normalized_question:
        terms["process"] = 10
        terms["am process"] = 15

    return terms


def normalize_text(text):
    """
    Make PDF text and user questions easier to compare.

    PDFs often contain special minus signs, long dashes, and degree symbols.
    """
    replacements = {
        "\u2212": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u00b0": " degrees ",
        "âˆ’": "-",
        "â€“": "-",
        "â€”": "-",
        "Â°": " degrees ",
    }

    for old_text, new_text in replacements.items():
        text = text.replace(old_text, new_text)

    return re.sub(r"\s+", " ", text.lower()).strip()


def remove_duplicate_chunks(chunks):
    """Remove repeated chunks while preserving order."""
    unique_chunks = []
    seen = set()

    for chunk in chunks:
        key = (
            chunk.metadata.get("source"),
            chunk.metadata.get("page"),
            chunk.page_content[:120],
        )

        if key not in seen:
            seen.add(key)
            unique_chunks.append(chunk)

    return unique_chunks


def build_context(retrieved_chunks):
    """Convert retrieved chunks into one context string for the LLM."""
    context_parts = []

    for index, chunk in enumerate(retrieved_chunks, start=1):
        page_number = format_page_number(chunk)
        context_parts.append(
            f"[Source {index}, page {page_number}]\n{chunk.page_content}"
        )

    return "\n\n".join(context_parts)


def format_page_number(chunk):
    """PyPDFLoader stores pages from 0, so add 1 for human page numbers."""
    page = chunk.metadata.get("page")
    return page + 1 if isinstance(page, int) else "unknown"


def create_llm():
    """Create the Gemini LLM used for the generation step."""
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Run python setup_key.py or create a .env file."
        )

    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=api_key,
        temperature=0,
    )


def generate_answer(llm, question, context):
    """Generate an answer using only the retrieved private document context."""
    prompt = f"""
You are an enterprise RAG assistant.

Rules:
1. Answer using only the context below.
2. If the answer is not in the context, say: "I don't know based on the provided document."
3. Keep the answer short and clear.
4. Mention the source page when possible.

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)
    return response.content


def answer_question(llm, vector_database, chunks, question):
    """Run the complete RAG flow for one user question."""
    retrieved_chunks = retrieve_top_chunks(vector_database, chunks, question)
    context = build_context(retrieved_chunks)
    answer = generate_answer(llm, question, context)

    print("\n" + "=" * 80)
    print(f"Question: {question}")
    print("-" * 80)
    print(f"Answer:\n{answer}")
    print("-" * 80)
    print("Top 3 retrieved chunks:")

    for index, chunk in enumerate(retrieved_chunks, start=1):
        page_number = format_page_number(chunk)
        preview = chunk.page_content[:220].replace("\n", " ")
        print(f"{index}. Page {page_number}: {preview}...")


def main():
    print("\nEnterprise RAG Project")
    print("=" * 80)

    print("\nPART 1: Data Preparation")
    chunks = prepare_data(PDF_PATH)

    print("\nPART 2: End-to-End RAG Pipeline")
    vector_database = build_vector_database(chunks)
    llm = create_llm()

    print(f"Vector database created with {len(chunks)} chunks.")
    print(f"Retriever will return the top {TOP_K} chunks.")
    print(f"LLM model: {GEMINI_MODEL}")

    while True:
        question = input("\nAsk a question about the PDF, or type 'exit': ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        if not question:
            continue

        answer_question(llm, vector_database, chunks, question)


if __name__ == "__main__":
    main()
