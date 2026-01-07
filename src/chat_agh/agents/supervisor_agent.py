from collections.abc import Generator
from typing import Any, Dict, List, Optional, Union

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, model_validator

from chat_agh.prompts import SUPERVISOR_AGENT_PROMPT_TEMPLATE
from chat_agh.utils.agents_info import RETRIEVAL_AGENTS, AgentDetails, AgentsInfo
from chat_agh.utils.chat_history import ChatHistory
from chat_agh.utils.model_inference import GoogleGenAIModelInference


class SupervisorOutput(BaseModel):
    retrieval_decision: bool
    response: str | None = None
    queries: Optional[dict[str, str]] = None

    @model_validator(mode="before")
    def check_fields_based_on_decision(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        decision = values.get("retrieval_decision")
        queries = values.get("queries")
        response = values.get("response")

        if decision:
            if not queries:
                raise ValueError(
                    "When retrieval_decision is True, 'queries' must be provided"
                )
            if response:
                raise ValueError(
                    "When retrieval_decision is True, 'answer' should not be provided"
                )
        else:
            if not response:
                raise ValueError(
                    "When retrieval_decision is False, 'answer' must be provided"
                )

            if queries is not None:
                raise ValueError(
                    "'queries' should not be provided when retrieval_decision is False"
                )

        agents_names = [agent.name for agent in RETRIEVAL_AGENTS]
        if queries:
            for agent in queries.keys():
                if agent not in agents_names:
                    raise ValueError(f"Agent '{agent}' is not defined")

        return values


class SupervisorAgent:
    def __init__(self) -> None:
        super().__init__()
        self.llm = GoogleGenAIModelInference()

        self.output_parser = PydanticOutputParser(pydantic_object=SupervisorOutput)
        self.prompt = PromptTemplate(
            input_variables=["agents_info", "chat_history", "latest_user_message"],
            template=SUPERVISOR_AGENT_PROMPT_TEMPLATE,
        )
        self.chain: Runnable[
            Dict[str, Union[Any, BaseMessage, list[BaseMessage]]], SupervisorOutput
        ] = self.prompt | self.llm | self.output_parser

    def invoke(
        self,
        agents_info: AgentsInfo,
        chat_history: ChatHistory,
        context: Optional[List[Document]] = None,
    ) -> SupervisorOutput:
        return self.chain.invoke(
            {
                "context": context,
                "agents_info": agents_info,
                "chat_history": chat_history[:-1],
                "latest_user_message": chat_history[-1].content,
            }
        )

    def stream(
        self, agents_info: AgentsInfo, chat_history: Any, context: Any
    ) -> Generator[Any, None, None]:
        start_state: Dict[str, Any] = {
            "context": context,
            "agents_info": agents_info,
            "chat_history": chat_history[:-1],
            "latest_user_message": chat_history[-1].content,
        }
        for chunk in self.chain.stream(start_state):
            yield chunk


if __name__ == "__main__":
    agent = SupervisorAgent()
    res = agent.invoke(
        agents_info=AgentsInfo(
            [
                AgentDetails(
                    name="recrutation_agent",
                    description="Agent retrieving informations about recrutation",
                    cached_history={
                        "query": "Jak zostać studentem AGH?",
                        "response": "Musisz przejsc proces rekrutacji",
                    },
                )
            ]
        ),
        chat_history=ChatHistory(
            messages=[
                HumanMessage("Hej"),
                AIMessage("Cześć!, Jak mogę ci pomóc?"),
                HumanMessage("Jak dostać się na AGH?"),
            ]
        ),
    )
    print(res)

    GoogleGenAIModelInference().get_usage()
