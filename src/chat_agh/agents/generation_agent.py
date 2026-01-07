from collections.abc import Generator
from typing import Any, Dict

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable

from chat_agh.prompts import GENERATION_PROMPT_TEMPLATE
from chat_agh.utils.model_inference import GoogleGenAIModelInference


class GenerationAgent:
    def __init__(self) -> None:
        super().__init__()
        self.llm = GoogleGenAIModelInference()

        self.prompt = PromptTemplate(
            input_variables=["agents_info", "chat_history"],
            template=GENERATION_PROMPT_TEMPLATE,
        )
        self.chain: Runnable[Dict[str, Any], Any] = self.prompt | self.llm

    def stream(self, chat_history: Any, context: Any) -> Generator[Any, None, None]:
        start_state: Dict[str, Any] = {
            "context": context,
            "chat_history": chat_history,
        }
        for chunk in self.chain.stream(start_state):
            yield chunk

    def invoke(self, chat_history: Any, context: Any) -> Any:
        start_state: Dict[str, Any] = {
            "context": context,
            "chat_history": chat_history,
        }
        return self.chain.invoke(start_state)
