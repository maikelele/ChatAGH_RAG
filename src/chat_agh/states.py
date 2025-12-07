from typing import TypedDict

from langchain_core.documents import Document

from chat_agh.utils.agents_info import AgentsInfo
from chat_agh.utils.chat_history import ChatHistory
from chat_agh.utils.retrieved_context import RetrievedContext


class ChatState(TypedDict, total=False):
    context: list[Document]
    chat_history: ChatHistory
    agents_info: AgentsInfo
    retrieval_decision: bool
    agents_queries: dict[str, str]
    response: str


class RetrievalStateRequired(TypedDict):
    query: str
    retrieved_context: list[RetrievedContext]


class RetrievalState(RetrievalStateRequired, total=False):
    retrieved_chunks: dict[str, list[Document]]
    summary: str
