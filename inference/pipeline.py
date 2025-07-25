import os
from fastapi import HTTPException
from preprocess.pipeline import preprocess_texts
from vector_store.retriever import query_vectorstore
from guardrails.guardrails_engine import apply_guardrails
from inference.predictor import generate_answer

# Store conversation history (limited to last 3 exchanges per chat_id)
user_history = {}


def run_rag_pipeline(query: str, lang: str, chat_id: str = "default", model_type: str = 'groq') -> str:
    """
    Run the RAG pipeline with conversation history for context-aware responses.
    Args:
        query (str): The user's question.
        lang (str): The language of the query ('english' or 'bengali').
        chat_id (str): Unique identifier for the conversation session.
        model_type (str): The type of model to use (hf or groq).
    Returns:
        str: The safe, generated response.
    """
    # Step 1: Preprocess
    cleaned_texts = preprocess_texts(query, lang=lang)

    path = f"vectorstore_{'en' if lang == 'english' else 'bn'}_semantic"

    if not os.path.exists(path):
        raise HTTPException(status_code=404,
                            detail=f"Vectorstore not found for lang={lang}. Please build first.")

    # Step 2: Query Vector Store
    docs = query_vectorstore(cleaned_texts, path)
    context = " ".join(docs)

    # Step 3: Get conversation history
    history = user_history.get(chat_id, [])

    # Step 4: Generate answer with history
    answer = generate_answer(query, context=context, model_type=model_type, history=history)

    # Step 5: Guardrails
    safe_answer = apply_guardrails(answer, lang=lang)

    # Step 6: Update conversation history (store last 3 exchanges)
    history.append({"query": query, "response": safe_answer})
    user_history[chat_id] = history[-3:]  # Keep only the last 3 exchanges
    return safe_answer
