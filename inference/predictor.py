from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langdetect import detect
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import os
from config.constants import MODEL_NAME, GROQ_MODEL


class Prediction:
    def __init__(self, model_name: str = MODEL_NAME, model_type: str = "hf"):
        """
        Initialize the Prediction class with the specified model type.
        Args:
            model_name (str): Name of the model (for Hugging Face or Groq).
            model_type (str): Type of model to use ("hf" for Hugging Face, "groq" for Groq).
        """
        self.model_name = model_name
        self.model_type = model_type.lower()

        if self.model_type == "hf":
            # Initialize Hugging Face model
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            # Use slow tokenizer to avoid tiktoken/SentencePiece conversion issues
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            self.generator = pipeline(
                task="text2text-generation",
                model=self.model,
                tokenizer=self.tokenizer
            )
        elif self.model_type == "groq":
            # Initialize Groq model via LangChain
            self.model = ChatGroq(
                temperature=0,
                groq_api_key=os.getenv('GROQ_API_KEY'),
                model=GROQ_MODEL,
            )
            # Create a prompt template for Groq with conversation history
            self.template = """You are an expert assistant for an educational chatbot, providing accurate and 
            professional responses based on the given context. Respond to the user's question in the same language as 
            the query, as detected (e.g., English or Bengali). Use the conversation history to maintain context and 
            ensure continuity. If the context contains relevant information, provide a concise and accurate answer. 
            If the context does not contain relevant information to answer the query, respond with: "Sorry, 
            I could not find relevant information to answer your question."

            Conversation History:
            {history}

            Question: {query}

            Context: {context}

            (NO PREAMBLE)"""
            self.prompt_template = PromptTemplate.from_template(self.template)
        else:
            raise ValueError("Invalid model_type. Use 'hf' for Hugging Face or 'groq' for Groq.")

    def create_prompt(self, query: str, context: str) -> str:
        """
        Create a prompt for the model.
        """
        return f"Answer the question using the context.\n\nQuestion: {query}\n\nContext: {context}"

    def generate(self, query: str, context: str, history: list = None) -> str:
        """
        Generate a response based on the query, context, and conversation history.
        Args:
            query (str): The user's question.
            context (str): The context to base the answer on.
            history (list): List of previous query-response pairs.
        Returns:
            str: The generated response.
        """
        if self.model_type == "hf":
            # Hugging Face model generation
            prompt = self.create_prompt(query, context)
            output = self.generator(prompt, max_new_tokens=256)[0]["generated_text"]
            return output
        else:
            # Groq model generation
            # Format history for the prompt
            formatted_history = ""
            if history:
                formatted_history = "\n".join(
                    f"User: {item['query']}\nAssistant: {item['response']}" for item in history
                )
            chain = self.prompt_template | self.model
            output = chain.invoke(input={'query': query, 'context': context, 'history': formatted_history})
            return output.content


# RAG-compatible wrapper
def generate_answer(query: str, context: str, model_type: str = "hf", model_name: str = MODEL_NAME,
                    history: list = None) -> str:
    """
    Generate an answer using the specified model type.
    Args:
        query (str): The user's question.
        context (str): The context to base the answer on.
        model_type (str): Type of model to use ("hf" for Hugging Face, "groq" for Groq).
        model_name (str): Name of the model to use.
        history (list): List of previous query-response pairs.
    Returns:
        str: The generated response.
    """
    model = Prediction(model_name=model_name, model_type=model_type)
    return model.generate(query, context, history)
