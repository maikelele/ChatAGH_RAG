from collections.abc import Generator
from typing import Any, cast

from langchain_core.messages import HumanMessage
from langgraph.graph.state import (  # type: ignore[attr-defined, unused-ignore]
    END,
    START,
    StateGraph,
)

from chat_agh.nodes import (
    GenerationNode,
    InitialRetrievalNode,
    RetrievalNode,
    SupervisorNode,
)
from chat_agh.states import ChatState
from chat_agh.utils import logger
from chat_agh.utils.agents_info import (
    RETRIEVAL_AGENTS,
    AgentDetails,
    AgentsInfo,
)
from chat_agh.utils.chat_history import ChatHistory


class ChatGraph:
    def __init__(self) -> None:
        self.graph = (
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
                    num_chunks=10,
                    k=5,
                ),
            )
            .add_node("supervisor_node", SupervisorNode())
            .add_node("retrieval_node", RetrievalNode())
            .add_node("generation_node", GenerationNode())
            .add_edge(START, "initial_retrieval_node")
            .add_edge("initial_retrieval_node", "supervisor_node")
            .add_conditional_edges(
                "supervisor_node",
                lambda state: (
                    "retrieval_node" if state["retrieval_decision"] else END
                ),
            )
            .add_edge("retrieval_node", "generation_node")
            .add_edge("generation_node", END)
            .compile()
        )

    def query(self, question: str) -> str:
        chat_history = ChatHistory(messages=[HumanMessage(question)])
        return self.invoke(chat_history)

    def invoke(
        self, chat_history: ChatHistory, *, config: dict[str, Any] | None = None
    ) -> str:
        result = self.invoke_with_details(chat_history, config=config)
        response = result.get("response")
        if not isinstance(response, str):
            raise TypeError("ChatGraph expected response to be a string")
        return response

    def stream(self, chat_history: ChatHistory) -> Generator[str, None, None]:
        state: ChatState = {
            "chat_history": chat_history,
            "agents_info": self._get_agents_info(),
        }
        for response_chunk in self.graph.stream(cast(Any, state), stream_mode="custom"):
            content = getattr(response_chunk, "content", None)
            if not isinstance(content, str):
                raise TypeError("ChatGraph stream yielded chunk without string content")
            yield content

    def invoke_with_details(
        self, chat_history: ChatHistory, *, config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        state: ChatState = {
            "chat_history": chat_history,
            "agents_info": self._get_agents_info(),
        }
        invoke_config = {} if config is None else config
        result = cast(
            dict[str, Any],
            self.graph.invoke(cast(Any, state), config=invoke_config),  # type: ignore[arg-type, unused-ignore]
        )
        return result

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

    chat_history = ChatHistory(
        messages=[HumanMessage("Kiedy zaczyna sie rekrutcja na AGH?")]
    )
    logger.info("START")

    res = chat_graph.invoke(
        chat_history, config={"configurable": {"generation_exec_mode": "invoke"}}
    )
    print(res)

    # for c in chat_graph.stream(chat_history):
    #     print(c)

    logger.info("END")

    from chat_agh.utils.model_inference import GoogleGenAIModelInference

    print(GoogleGenAIModelInference().get_usage())
