import logging
import os
from typing import Any

from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

from chat_agh.utils.consts import EMBEDDINGS_MODEL, MODELS_WITH_RPM

load_dotenv()

mongo_client: MongoClient[Any] = MongoClient(
    os.environ["MONGODB_URI"], tlsAllowInvalidCertificates=True
)

embedding_model = SentenceTransformer(
    EMBEDDINGS_MODEL,
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

model_draw_counts: dict[str, int] = {name: 0 for name in list(MODELS_WITH_RPM.keys())}
