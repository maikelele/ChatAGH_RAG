from collections.abc import Generator
from typing import Any, cast
from chat_agh.utils.utils import logger

from langchain_core.messages import HumanMessage
from langgraph.graph.state import END, START, StateGraph

from chat_agh.nodes import (
    InitialRetrievalNode,
    RetrievalNode,
    SupervisorNode,
)
from chat_agh.states import ChatState
from chat_agh.utils.agents_info import (
    RETRIEVAL_AGENTS,
    AgentDetails,
    AgentsInfo,
)
from chat_agh.utils.chat_history import ChatHistory


class ChatGraph:
    def __init__(self) -> None:
        self.graph: Any = (
            StateGraph(ChatState)
            .add_node(
                "initial_retrieval_node",
                InitialRetrievalNode(
                    collections=[
                        "cluster_0",
                        "cluster_6",
                        "cluster_7",
                        "cluster_8",
                        "cluster_9",
                    ],
                    num_chunks=5,
                    k=5,
                ),
            )
            .add_node("supervisor_node", SupervisorNode())
            .add_node("retrieval_node", RetrievalNode())
            .add_edge(START, "initial_retrieval_node")
            .add_edge("initial_retrieval_node", "supervisor_node")
            .add_conditional_edges(
                "supervisor_node",
                lambda state: (
                    "retrieval_node" if state["retrieval_decision"] else END
                ),
            )
            .add_edge("retrieval_node", "supervisor_node")
            .compile()
        )

    def query(self, question: str) -> str:
        chat_history = ChatHistory(messages=[HumanMessage(question)])
        return self.invoke(chat_history)

    def invoke(
        self, chat_history: ChatHistory, config: dict[Any, Any] | None = None
    ) -> str:
        state = ChatState(
            chat_history=chat_history,
            agents_info=self._get_agents_info(),
            num_search_tries=0,
        )
        result = cast(dict[str, Any], self.graph.invoke(state, config=config))
        response = result.get("response")
        if not isinstance(response, str):
            raise TypeError("ChatGraph expected response to be a string")
        return response

    def stream(self, chat_history: ChatHistory) -> Generator[str, None, None]:
        state = ChatState(
            chat_history=chat_history,
            agents_info=self._get_agents_info(),
            num_search_tries=0,
        )
        for response_chunk in self.graph.stream(state, stream_mode="custom"):
            content = getattr(response_chunk, "content", None)
            if not isinstance(content, str):
                raise TypeError("ChatGraph stream yielded chunk without string content")
            yield content

    def _get_agents_info(self) -> AgentsInfo:
        return AgentsInfo(
            agents_details=[
                AgentDetails(
                    name=agents_details.name,
                    description=agents_details.description,
                    cached_history=None,
                )
                for agents_details in RETRIEVAL_AGENTS
            ]
        )


if __name__ == "__main__":
    chat_graph = ChatGraph()

    chat_history = ChatHistory(messages=[HumanMessage("Kiedy jest otwarcie wydziału?")])
    logger.info("START")

    res = chat_graph.invoke(
        chat_history, config={"configurable": {"generation_exec_mode": "invoke"}}
    )
    print(res)

    # for c in chat_graph.stream(chat_history):
    #     print(c)

    logger.info("END")

    from chat_agh.utils.model_inference import OpenAIModelInference

    print(OpenAIModelInference().get_usage())
