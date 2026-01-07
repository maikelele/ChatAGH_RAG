from typing import Any, Callable, Optional

from google.api_core.exceptions import ResourceExhausted
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI

from chat_agh.utils.consts import MODELS_WITH_RPM
from chat_agh.utils.singletons import logger, model_draw_counts
from chat_agh.utils.utils import (
    GEMINI_API_KEYS,
    draw_from_list,
    gemini_api_key_draw_counts,
)


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

    def __init__(self) -> None:
        self.model = self._choose_model()
        self.api_key = self._choose_api_key()
        llm = ChatGoogleGenerativeAI(model=self.model, api_key=self.api_key)
        super().__init__(llm=llm)

    def invoke(
        self, input: Any, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Any:
        out = self._call_with_model_fallback(
            lambda: self.llm.invoke(input, config=config)
        )
        self._update_usage(out)
        return out

    def stream(
        self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Any:
        yield from self._call_with_model_fallback(
            lambda: self.llm.stream(input, config=config), is_stream=True
        )

    def _choose_model(self) -> str:
        return draw_from_list(
            candidates_with_weights=MODELS_WITH_RPM, draw_counts=model_draw_counts
        )

    def _choose_api_key(self) -> str:
        return draw_from_list(
            candidates_with_weights={key: 1 for key in GEMINI_API_KEYS},
            draw_counts=gemini_api_key_draw_counts,
        )

    def _swap_model(self) -> None:
        self.model = self._choose_model()
        self.api_key = self._choose_api_key()
        logger.warning(
            "Switching to Gemini model %s and rotating API key due to rate limits",
            self.model,
        )
        self.llm = ChatGoogleGenerativeAI(model=self.model, api_key=self.api_key)

    def _call_with_model_fallback(
        self, caller: Callable[[], Any], *, is_stream: bool = False
    ) -> Any:
        attempts = max(1, len(MODELS_WITH_RPM))
        last_exc: ResourceExhausted | None = None
        for attempt in range(attempts):
            try:
                result = caller()
                if not is_stream:
                    return result
                return self._consume_stream(result)
            except ResourceExhausted as exc:
                last_exc = exc
                current_model = getattr(self, "model", "unknown")
                logger.warning(
                    "Gemini model %s hit rate limits (attempt %s/%s): %s",
                    current_model,
                    attempt + 1,
                    attempts,
                    exc,
                )
                if attempt == attempts - 1:
                    break
                self._swap_model()
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Model fallback failed without raising ResourceExhausted")

    def _consume_stream(self, stream_result: Any) -> Any:
        for chunk in stream_result:
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
