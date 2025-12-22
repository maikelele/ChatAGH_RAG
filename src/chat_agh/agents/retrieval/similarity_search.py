from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, Tuple

from langchain_core.documents import Document
from pymongo.collection import Collection

from chat_agh.agents.retrieval.utils import aggregate_by_url
from chat_agh.states import RetrievalState
from chat_agh.utils.utils import (
    MONGO_DATABASE_NAME,
    log_execution_time,
    logger,
    mongo_client,
)
from chat_agh.utils.retrieved_context import RetrievedContext
from chat_agh.vector_store.mongodb import MongoDBVectorStore


class SimilaritySearch:
    def __init__(
        self, index_name: str, num_retrieved_chunks: int = 5, window_size: int = 1
    ) -> None:
        self.num_retrieved_chunks = num_retrieved_chunks
        self.window_size = window_size
        self.vector_store = MongoDBVectorStore(index_name)
        self.collection: Collection[Dict[str, Any]] = mongo_client[MONGO_DATABASE_NAME][
            index_name
        ]

    @log_execution_time
    def __call__(
        self, state: RetrievalState
    ) -> Dict[str, Dict[str, list[Document]] | list[RetrievedContext]]:
        retrieved_chunks = self.vector_store.search(
            state["query"], final_limit=self.num_retrieved_chunks
        )

        aggregated_docs = aggregate_by_url(retrieved_chunks)
        logger.info(
            "Retrieved {} documents, source urls: {}".format(
                len(retrieved_chunks), aggregated_docs.keys()
            )
        )
        chunks_windows = self.get_chunks_windows(aggregated_docs)

        return {
            "retrieved_chunks": chunks_windows,
            "retrieved_context": [
                RetrievedContext(source_url=url, chunks=chunks, related_chunks={})
                for url, chunks in chunks_windows.items()
            ],
        }

    def get_chunks_windows(
        self, urls: Dict[str, list[Document]]
    ) -> Dict[str, list[Document]]:
        """Returns chunks for specific sequence_numbers per URL (batched and deduplicated)."""
        retrieved_docs: Dict[str, list[Document]] = {}

        def process_url(url: str, docs: list[Document]) -> Tuple[str, list[Document]]:
            seq_numbers: set[int] = set()
            for doc in docs:
                seq = doc.metadata["sequence_number"]
                window_range = range(
                    max(seq - self.window_size, 0), seq + self.window_size + 1
                )
                seq_numbers.update(window_range)

            query = {
                "metadata.url": url,
                "metadata.sequence_number": {"$in": list(seq_numbers)},
            }
            results: Iterable[Dict[str, Any]] = self.collection.find(query)

            seen: set[Tuple[str, int]] = set()
            unique_docs_raw: list[Dict[str, Any]] = []
            for d in results:
                key = (d["metadata"]["url"], d["metadata"]["sequence_number"])
                if key not in seen:
                    seen.add(key)
                    unique_docs_raw.append(d)

            unique_docs = [
                Document(page_content=d["text"], metadata=d["metadata"])
                for d in unique_docs_raw
            ]

            return url, sorted(unique_docs, key=lambda d: d.metadata["sequence_number"])

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_url, url, docs) for url, docs in urls.items()
            ]
            for future in as_completed(futures):
                url, docs = future.result()
                retrieved_docs[url] = docs

        return retrieved_docs
