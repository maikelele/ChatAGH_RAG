import logging
import os

from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

load_dotenv()

mongo_client: MongoClient = MongoClient(
    os.environ["MONGODB_URI"], tlsAllowInvalidCertificates=True
)

# Force the embedding model to run on CPU so that any limited GPU memory
# can be fully utilized by Ollama during local experiments.
embedding_model: SentenceTransformer = SentenceTransformer(
    "intfloat/multilingual-e5-large",
    device="cpu",
)

logger = logging.getLogger("chat_graph_logger")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
