from typing import Any, Dict, Iterator, List, Optional

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.types import StreamWriter

from chat_agh.agents import GenerationAgent
from chat_agh.states import ChatState
from chat_agh.utils.utils import log_execution_time, retry_on_exception


def _extract_text(payload: Any) -> Any:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("content"), str):
        return payload["content"]
    content = getattr(payload, "content", None)
    if isinstance(content, str):
        return content
    raise TypeError("GenerationAgent returned payload without string content")


def _build_context(state: ChatState) -> str:
    if any(agent.cached_history for agent in state["agents_info"].agents_details):
        return str(state["agents_info"])
    docs: List[Document] = state["context"]
    return "\n".join(d.page_content for d in docs)


class GenerationNode:
    def __init__(self) -> None:
        self.agent = GenerationAgent()

    def invoke(self, state: ChatState) -> Dict[str, str]:
        args = {"context": _build_context(state), "chat_history": state["chat_history"]}
        result = self.agent.invoke(**args)
        return {"response": _extract_text(result)}

    def stream(self, state: ChatState) -> Iterator[Dict[str, str]]:
        args = {"context": _build_context(state), "chat_history": state["chat_history"]}
        for chunk in self.agent.stream(**args):
            yield {"response": _extract_text(chunk)}

    @retry_on_exception(attempts=2, delay=1, backoff=3)
    @log_execution_time
    def __call__(
        self,
        state: ChatState,
        *,
        writer: Optional[StreamWriter] = None,
        config: RunnableConfig,
    ) -> Dict[str, str]:
        args = {"context": _build_context(state), "chat_history": state["chat_history"]}

        cfg = config.get("configurable") or {}
        mode = cfg.get("generation_exec_mode", "stream")

        if mode == "invoke":
            final_text = self.agent.invoke(**args)
            return {"response": _extract_text(final_text)}

        elif mode == "stream":
            writer = get_stream_writer()
            final_text = ""
            for chunk in self.agent.stream(**args):
                writer(chunk)
                final_text += _extract_text(chunk)
            return {"response": final_text}

        else:
            raise ValueError(f"Unknown generation_exec_mode '{mode}'")
