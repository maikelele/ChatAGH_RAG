from typing import Any, Dict

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable

from chat_agh.prompts import SUMMARY_GENERATION_PROMPT_TEMPLATE
from chat_agh.states import RetrievalState
from chat_agh.utils.retrieved_context import RetrievedContext
from chat_agh.utils.utils import log_execution_time, retry_on_exception
from chat_agh.utils.model_inference import GoogleGenAIModelInference


class SummaryGeneration:
    def __init__(self) -> None:
        self.llm = GoogleGenAIModelInference()
        self.prompt = PromptTemplate(
            input_variables=["context", "query"],
            template=SUMMARY_GENERATION_PROMPT_TEMPLATE,
        )
        self.chain: Runnable[Dict[str, Any], Any] = self.prompt | self.llm

    @log_execution_time
    @retry_on_exception(attempts=3, delay=1, backoff=3)
    def __call__(self, state: RetrievalState) -> Dict[str, Any]:
        context = ""
        retrieved_contexts: list[RetrievedContext] = state["retrieved_context"]
        for retrieved_context in retrieved_contexts:
            context += "## CONTEXT\n" + retrieved_context.text + "\n"

        summary = self.chain.invoke({"context": context, "query": state["query"]})
        return {"summary": summary}
