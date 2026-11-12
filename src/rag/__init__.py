from dotenv import load_dotenv

load_dotenv()

from .medium_analyser import main as medium_analyser
from .medium_ingestion import main as medium_ingestion
from .create_new_index import main as create_new_index

__all__ = [
    "medium_analyser",
    "medium_ingestion",
    "create_new_index",
]
