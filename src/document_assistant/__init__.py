from dotenv import load_dotenv

load_dotenv()

from .main import main as document_assistant
from .ingestion import main as document_ingestion

__all__ = [
    "document_assistant",
    "document_ingestion",
]
