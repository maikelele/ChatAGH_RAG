from __future__ import annotations

from typing import Any

from opik import track

from chat_agh.agents import GenerationAgent
from chat_agh.utils.singletons import logger
from chat_agh.vector_store.mongodb import MongoDBVectorStore
from scripts.evaluation_tasks.base import CurrentSearchParameters


class VectorSearchEvaluationTask:
    """Run an evaluation round with custom MongoDB vector search parameters."""

    def __init__(
        self,
        *,
        collection_names: list[str],
        search_setting: CurrentSearchParameters,
        distance_metric: str = "cosine",
        vector_index_name: str = "vector_index",
        search_index_name: str = "default",
    ) -> None:
        self._collection_names = collection_names
        self._search_setting = search_setting
        self._distance_metric = distance_metric
        self._vector_stores = [
            MongoDBVectorStore(
                collection_name=collection_name,
                vector_index_name=vector_index_name,
                search_index_name=search_index_name,
                similarity=distance_metric,
                create_indexes=False,
            )
            for collection_name in self._collection_names
        ]
        self._generation_agent = GenerationAgent()

        settings_descriptor = "_".join(
            f"{key}-{value}" for key, value in self._search_setting.to_dict().items()
        )
        self.task_name = settings_descriptor.replace(" ", "-")

        self.experiment_config = {
            "collection": self._collection_names,
            "similarity": distance_metric,
            **search_setting.to_dict(),
            "distance_metric": self._distance_metric,
        }
        self._similarity = distance_metric

    @track(name="vector_search_run()")
    def run(self, input_data: dict[str, str]) -> dict[str, Any]:
        question = input_data.get("question")
        if not question:
            raise ValueError("Dataset entry does not contain a 'question' field")
        logger.info(
            "Executing vector search on '%s' with params %s",
            self._collection_names,
            self._search_setting.to_dict(),
        )

        documents = self._search_setting.run(self._vector_stores, question)

        serialized_context = [
            {
                "url": doc.metadata.get("url"),
                "score": doc.metadata.get("score"),
                "sequence_number": doc.metadata.get("sequence_number"),
            }
            for doc in documents
        ]

        retrieved_contexts = [doc.page_content for doc in documents]

        response = self._generate_response(question, retrieved_contexts)

        return {
            "question": question,
            "response": response,
            "retrieved_contexts": retrieved_contexts,
            "retrieved_context_metadata": serialized_context,
            "search_parameters": self._search_setting.to_dict(),
            "distance_metric": self._distance_metric,
            "similarity": self._similarity,
        }

    def _generate_response(self, question: str, contexts: list[str]) -> str:
        context_blob = "\n".join(contexts) if contexts else ""
        chat_history = f"USER: {question}"
        try:
            raw_response = self._generation_agent.invoke(
                context=context_blob, chat_history=chat_history
            )
        except Exception:
            logger.exception(
                "GenerationAgent failed to build response for evaluation question"
            )
            return context_blob

        if isinstance(raw_response, str):
            return raw_response
        if isinstance(raw_response, dict):
            content = raw_response.get("content")
            if isinstance(content, str):
                return content
            return str(raw_response)

        content_attr = getattr(raw_response, "content", None)
        if isinstance(content_attr, str):
            return content_attr

        return str(raw_response)
