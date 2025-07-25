# RAG-Based Chatbot with FastAPI

This project implements a Retrieval-Augmented Generation (RAG) based chatbot using Python and FastAPI. The chatbot leverages vector embeddings and a retrieval system to provide contextually relevant responses.

## Prerequisites

- Python 3.9 or higher
- pip package manager
- (Optional) virtualenv or conda for virtual environment management

## Setup Instructions

### 1. Create and Activate Virtual Environment

```bash
# Create virtual environment
python -m venv rag_chatbot_env

# Activate the environment
# On Windows:
rag_chatbot_env\Scripts\activate
# On Unix or MacOS:
source rag_chatbot_env/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
Create a .env file in the root directory with your API keys and configurations:
```bash
HUGGINGFACE_TOKEN=huggingface_key
GROQ_API_KEY=groq_key
```

### 4. Run the FastAPI Application
```bash
uvicorn main:app --reload
```

The application will be available at:
- Local: http://127.0.0.1:8000
- Docs: http://127.0.0.1:8000/docs

## File Structure

```bash
├── config/
│ ├── settings.yaml # Configuration file
│ └── constants.py # Project constants
├── data/
│ ├── raw/ # Raw documents
│ │ ├── bangla/ # Bangla documents
│ │ └── english/ # English documents
│ ├── processed/ # Processed documents
│ │ ├── bangla/
│ │ └── english/
│ └── split/ # Split documents
│ ├── bangla/
│ └── english/
├── preprocess/ # Preprocessing modules
│ ├── cleaner.py # Text cleaning
│ ├── language_detector.py # Language detection
│ ├── pipeline.py # Preprocessing pipeline
│ └── init.py
├── splitter/ # Text splitting
│ ├── text_splitter.py # Document splitting
│ ├── splitter_utils.py # Splitter utilities
│ └── init.py
├── vector_store/ # Vector storage
│ ├── embedder.py # Embedding generation
│ ├── store.py # Vector store
│ ├── retriever.py # Retrieval logic
│ └── init.py
├── finetune/ # Model fine-tuning
│ ├── prepare_dataset.py # Dataset prep
│ ├── train_model.py # Training
│ ├── evaluation.py # Model evaluation
│ └── init.py
├── guardrails/ # Safety checks
│ ├── profanity_filter.py # Content filtering
│ ├── hallucination_checker.py # Fact verification
│ ├── guardrails_engine.py # Main guardrails
│ └── init.py
├── inference/ # Inference
│ ├── predictor.py # Prediction
│ ├── pipeline.py # Inference pipeline
│ ├── api.py # FastAPI endpoints
│ └── init.py
├── notebooks/ # Jupyter notebooks
│ ├── exploration.ipynb # Data exploration
│ └── debug.ipynb # Debugging
├── scripts/ # Utility scripts
│ ├── preprocess_all.py # Bulk preprocessing
│ ├── build_vectorstore.py # Vector store builder
│ ├── run_finetune.py # Fine-tuning
│ ├── run_inference.py # Inference runner
│ └── test_guardrails.py # Guardrails testing
├── tests/ # Unit tests
│ ├── test_preprocess.py
│ ├── test_splitter.py
│ ├── test_vector_store.py
│ └── test_predictor.py
├── requirements.txt # Python dependencies
├── README.md # This file
└── .env # Environment variables
```
