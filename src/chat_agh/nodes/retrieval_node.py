from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

from chat_agh.agents import RetrievalAgent
from chat_agh.states import ChatState
from chat_agh.utils import (
    RETRIEVAL_AGENTS,
    AgentDetails,
    AgentsInfo,
    log_execution_time,
    logger,
)


class RetrievalNode:
    def __init__(self) -> None:
        self.retrieval_agents: List[RetrievalAgent] = [
            RetrievalAgent(
                agent_name=agent_info.name,
                index_name=agent_info.vector_store_index_name,
                description=agent_info.description,
                graph_augmentation=False,
                num_retrieved_chunks=4,
                window_size=1,
            )
            for agent_info in RETRIEVAL_AGENTS
        ]

    @log_execution_time
    def __call__(self, state: ChatState) -> Dict[str, AgentsInfo | str]:
        queries = state["agents_queries"]
        responses: List[AgentDetails] = []

        def query_agent(agent: RetrievalAgent, query: str) -> AgentDetails:
            logger.info(f"Querying agent: {agent.name}. Query: {query}")
            agent_response = agent.query(query)
            logger.info(f"Agent response: {agent_response}")
            return AgentDetails(
                name=agent.name,
                description=agent.description,
                cached_history={
                    "query": query,
                    "response": agent_response,
                },
            )

        futures = []
        with ThreadPoolExecutor() as executor:
            for agent_name, query in queries.items():
                for agent in self.retrieval_agents:
                    if agent.name == agent_name:
                        futures.append(executor.submit(query_agent, agent, query))

            for future in as_completed(futures):
                responses.append(future.result())

        return {
            "agents_info": AgentsInfo(agents_details=responses),
            "context": "\n".join(str(r) for r in responses),
        }
