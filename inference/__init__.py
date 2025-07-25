from dotenv import load_dotenv
import os
from huggingface_hub import login

# Load HuggingFace token from .env
load_dotenv()
token = os.getenv("HUGGINGFACE_TOKEN")
if token:
    login(token=token)
