from typing import Dict, Optional

from langchain_core.messages import AIMessage
from langgraph.config import get_stream_writer
from langgraph.types import StreamWriter

from chat_agh.agents import SupervisorAgent
from chat_agh.states import ChatState
from chat_agh.utils.singletons import logger
from chat_agh.utils.utils import log_execution_time, retry_on_exception


class SupervisorNode:
    def __init__(self) -> None:
        self.agent = SupervisorAgent()

    @retry_on_exception(attempts=2, delay=1, backoff=3)
    @log_execution_time
    def __call__(
        self,
        state: ChatState,
        writer: Optional[StreamWriter] = None,
    ) -> Dict[str, Optional[str | Dict[str, str]] | bool]:
        agent_response = self.agent.invoke(
            agents_info=state["agents_info"],
            chat_history=state["chat_history"],
            context=state["context"],
        )

        logger.info(f"Retrieval decision: {agent_response.retrieval_decision}")
        if agent_response.retrieval_decision:
            logger.info(f"Queries: {agent_response.queries}")
        else:
            if agent_response.response:
                logger.info(f"Response: {agent_response.response}")
                writer = get_stream_writer()
                writer(AIMessage(agent_response.response))

        return {
            "retrieval_decision": agent_response.retrieval_decision,
            "agents_queries": agent_response.queries,
            "response": agent_response.response,
        }
