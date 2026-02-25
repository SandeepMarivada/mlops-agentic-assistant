import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/mistral-7b-instruct")
CHROMA_PERSIST_DIR = "./chroma_db"
DATA_DIR = "./data"
