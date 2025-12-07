from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from typing import cast

from langchain_core.documents import Document

from chat_agh.states import ChatState
from chat_agh.utils import log_execution_time, logger
from chat_agh.vector_store.mongodb import MongoDBVectorStore


class InitialRetrievalNode:
    def __init__(self, collections: list[str], num_chunks: int = 5, k: int = 5):
        self.num_chunks = num_chunks
        self.vector_stores = [
            MongoDBVectorStore(collection) for collection in collections
        ]
        self.k = k

    @log_execution_time
    def __call__(self, state: ChatState) -> dict[str, list[Document]]:
        chat_history = state["chat_history"]
        chat_history_text = "".join(
            [cast(str, message.content) for message in chat_history.messages]
        )

        if len(chat_history_text) > 20:
            logger.info(f"Initial retrieval, query: {chat_history_text}")
            with ThreadPoolExecutor() as executor:
                results = list(
                    chain.from_iterable(
                        executor.map(
                            lambda vs: vs.search(chat_history_text, k=self.k),
                            self.vector_stores,
                        )
                    )
                )
            final_result = self._reranking(results)
            logger.info(
                f"Found {len(results)} chunks in"
                f" {len(self.vector_stores)} collections,"
                f" reranked to: {len(final_result)}"
            )

            return {"context": final_result}
        else:
            logger.info(f"Initial retrieval skipped: {chat_history_text}")
            return {"context": []}

    def _reranking(self, results: list[Document]) -> list[Document]:
        results = sorted(results, key=lambda r: r.metadata["score"], reverse=True)
        return results[: self.num_chunks]
