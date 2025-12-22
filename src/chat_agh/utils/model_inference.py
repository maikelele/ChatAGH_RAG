import os
import json
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI


DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_GOOGLE_GEN_AI_MODEL = "gemini-2.5-flash"


def approx_tokens(text: str) -> int:
    return max(1, int(len(text) / 4))


class ModelInference(Runnable[Any, Any]):
    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm

    def invoke(
        self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Any:
        result = self.llm.invoke(input, config=config)
        return result


class GoogleGenAIModelInference(ModelInference):
    total_calls = 0
    total_tokens = 0
    total_input_tokens = 0
    total_output_tokens = 0

    def __init__(self, model: str = DEFAULT_GOOGLE_GEN_AI_MODEL) -> None:
        self.api_keys = json.loads(os.getenv("GEMINI_API_KEYS", "[]"))
        if not self.api_keys:
            raise RuntimeError("GEMINI_API_KEYS is empty or not set.")
        self.model = model

        llm = ChatGoogleGenerativeAI(
            model=model,
            api_key=self.api_keys[0],
        )
        super().__init__(llm=llm)

    def invoke(
        self, input: Any, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Any:
        out = self.llm.invoke(input, config=config)
        self._update_usage(out)
        return out

    def stream(
        self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Any:
        for chunk in self.llm.stream(input, config=config):
            yield chunk

    def _update_usage(self, out: Any) -> Any:
        GoogleGenAIModelInference.total_calls += 1
        GoogleGenAIModelInference.total_output_tokens += out.usage_metadata[
            "output_tokens"
        ]
        GoogleGenAIModelInference.total_input_tokens += out.usage_metadata[
            "input_tokens"
        ]
        GoogleGenAIModelInference.total_tokens += out.usage_metadata["total_tokens"]

    @classmethod
    def get_usage(cls) -> dict[str, int]:
        return {
            "calls": cls.total_calls,
            "total_tokens": cls.total_tokens,
            "total_input_tokens": cls.total_input_tokens,
            "total_output_tokens": cls.total_output_tokens,
        }


class OpenAIModelInference(ModelInference):
    total_calls = 0
    total_tokens = 0
    total_input_tokens = 0
    total_output_tokens = 0

    def __init__(
        self,
        model: str = DEFAULT_OPENAI_MODEL,
        temperature: float = 0.0,
        timeout: Optional[float] = None,
    ) -> None:
        api_keys = json.loads(os.getenv("OPENAI_API_KEYS", "[]") or "[]")
        single_key = os.getenv("OPENAI_API_KEY")

        if not api_keys and not single_key:
            raise RuntimeError(
                "OPENAI_API_KEYS is empty/not set and OPENAI_API_KEY is not set."
            )

        self.api_keys = api_keys or [single_key]
        self.model = model

        llm = ChatOpenAI(
            model=model,
            api_key=self.api_keys[0],
            temperature=temperature,
            timeout=timeout,
        )
        super().__init__(llm=llm)

    def invoke(
        self, input: Any, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Any:
        out = self.llm.invoke(input, config=config, **kwargs)
        self._update_usage(out)
        return out

    def stream(
        self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Any:
        for chunk in self.llm.stream(input, config=config, **kwargs):
            yield chunk

    def _update_usage(self, out: Any) -> None:
        OpenAIModelInference.total_calls += 1

        usage: dict[Any, Any] = {}
        if hasattr(out, "response_metadata") and isinstance(
            out.response_metadata, dict
        ):
            usage = out.response_metadata.get("token_usage", {}) or {}

        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        total_tokens = int(
            usage.get("total_tokens", prompt_tokens + completion_tokens) or 0
        )

        OpenAIModelInference.total_input_tokens += prompt_tokens
        OpenAIModelInference.total_output_tokens += completion_tokens
        OpenAIModelInference.total_tokens += total_tokens

    @classmethod
    def get_usage(cls) -> dict[str, int]:
        return {
            "calls": cls.total_calls,
            "total_tokens": cls.total_tokens,
            "total_input_tokens": cls.total_input_tokens,
            "total_output_tokens": cls.total_output_tokens,
        }
