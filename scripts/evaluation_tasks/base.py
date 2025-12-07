from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

from langchain_core.documents import Document

from chat_agh.vector_store.mongodb import MongoDBVectorStore

SearchMode = Literal["dense", "lexical", "hybrid_rrf"]


class CurrentSearchParameters:
    """Current production parameters used as defaults for experiments."""

    def __init__(self) -> None:
        self.mode: SearchMode = "hybrid_rrf"
        self.final_limit: int = 5
        self.lexical_limit: int = 10
        self.fuzzy: bool = False
        self.fuzzy_max_edits: int = 2
        self.fuzzy_prefix_length: int = 0
        self.dense_limit: int = 10
        self.num_dense_candidates: int = 200
        self.text_weight: float = 0.5
        self.vector_weight: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"mode": self.mode}
        match self.mode:
            case "lexical":
                if self.fuzzy:
                    if self.fuzzy_max_edits is not None:
                        payload["fuzzy_max_edits"] = self.fuzzy_max_edits
                    if self.fuzzy_prefix_length is not None:
                        payload["fuzzy_prefix_length"] = self.fuzzy_prefix_length
                payload["lexical_limit"] = self.lexical_limit
            case "dense":
                payload["dense_limit"] = self.dense_limit
                payload["num_dense_candidates"] = self.num_dense_candidates
            case "hybrid_rrf":
                payload["text_weight"] = self.text_weight
                payload["vector_weight"] = self.vector_weight
                payload["final_limit"] = self.final_limit

        return payload

    def _search_with_current_parameters(
        self, vector_stores: list[MongoDBVectorStore], query: str
    ) -> list[Document]:
        with ThreadPoolExecutor() as executor:
            per_store_results = list(
                executor.map(
                    lambda vs: vs.search(
                        query=query,
                        mode=self.mode,
                        lexical_limit=self.lexical_limit,
                        fuzzy=self.fuzzy,
                        fuzzy_max_edits=self.fuzzy_max_edits,
                        fuzzy_prefix_length=self.fuzzy_prefix_length,
                        dense_limit=self.dense_limit,
                        num_dense_candidates=self.num_dense_candidates,
                        text_weight=self.text_weight,
                        vector_weight=self.vector_weight,
                        final_limit=self.final_limit,
                    ),
                    vector_stores,
                )
            )

        results: list[Document] = [
            document
            for store_results in per_store_results
            for document in store_results
        ]
        return self._reranking(results, self.final_limit)

    def _reranking(self, results: list[Document], limit: int) -> list[Document]:
        results = sorted(results, key=lambda r: r.metadata["score"], reverse=True)
        return results[:limit]

    def run(
        self, vector_stores: list[MongoDBVectorStore], query: str
    ) -> list[Document]:
        raise NotImplementedError("Subclasses must implement run().")


class SearchParametersOverride(CurrentSearchParameters):
    """Utility search parameters override used for experiment sweeps."""

    _OVERRIDABLE_FIELDS = {
        "mode",
        "final_limit",
        "lexical_limit",
        "fuzzy",
        "fuzzy_max_edit",
        "fuzzy_prefix_length",
        "dense_limit",
        "num_dense_candidates",
        "vector_weight",
        "text_weight",
    }

    def __init__(self, **overrides: Any) -> None:
        super().__init__()
        for field_name, value in overrides.items():
            if field_name not in self._OVERRIDABLE_FIELDS:
                msg = f"Unsupported search parameter override: {field_name}"
                raise ValueError(msg)
            setattr(self, field_name, value)

    def run(
        self, vector_stores: list[MongoDBVectorStore], query: str
    ) -> list[Document]:
        return self._search_with_current_parameters(vector_stores, query)
