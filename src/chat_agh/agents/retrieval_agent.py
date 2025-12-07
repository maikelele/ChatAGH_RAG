from typing import Any, cast

from langgraph.graph.state import StateGraph

from chat_agh.agents.retrieval import (
    ContextRetrieval,
    SimilaritySearch,
    SummaryGeneration,
)
from chat_agh.states import RetrievalState


class RetrievalAgent:
    def __init__(
        self,
        agent_name: str,
        index_name: str,
        description: str,
        num_retrieved_chunks: int = 5,
        num_context_chunks: int = 3,
        window_size: int = 0,
    ) -> None:
        self.name = agent_name
        self.index_name = index_name
        self.description = description
        self.graph = (
            StateGraph(RetrievalState)
            .add_node(
                "similarity_search",
                SimilaritySearch(
                    index_name=self.index_name,
                    num_retrieved_chunks=num_retrieved_chunks,
                    window_size=window_size,
                ),
            )
            .add_node(
                "context_retrieval",
                ContextRetrieval(index_name=index_name, num_chunks=num_context_chunks),
            )
            .add_node("summary_generation", SummaryGeneration())
            .add_edge("similarity_search", "context_retrieval")
            .add_edge("context_retrieval", "summary_generation")
            .set_entry_point("similarity_search")
            .compile()
        )

    def query(self, query: str) -> str:
        state: RetrievalState = {"query": query, "retrieved_context": []}
        result = cast(dict[str, Any], self.graph.invoke(cast(Any, state)))
        summary = result.get("summary")
        if summary is None:
            msg = "Retrieval graph result is missing the summary"
            raise TypeError(msg)
        content = getattr(summary, "content", None)
        if not isinstance(content, str):
            msg = "Retrieval graph summary is missing string content"
            raise TypeError(msg)
        return content
