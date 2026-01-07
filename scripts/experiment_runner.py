from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, TypeVar, cast

import litellm
import yaml  # type: ignore[import-untyped, unused-ignore]
from google.api_core.exceptions import ResourceExhausted
from langchain_core.callbacks.base import Callbacks
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import LLMResult
from langchain_core.prompt_values import PromptValue
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from opik import Opik, evaluate
from opik.evaluation.evaluation_result import EvaluationResult
from opik.evaluation.metrics import BaseMetric
from opik.integrations.langchain import OpikTracer
from ragas.llms import BaseRagasLLM, LangchainLLMWrapper
from ragas.metrics import (
    ContextRecall,
    Faithfulness,
    LLMContextPrecisionWithReference,
)

from chat_agh.utils.consts import MODELS_WITH_RPM, OLLAMA_MODEL
from chat_agh.utils.singletons import logger, model_draw_counts
from chat_agh.utils.utils import (
    GEMINI_API_KEYS,
    draw_from_list,
    gemini_api_key_draw_counts,
)
from scripts.consts import PROJECT_NAME
from scripts.evaluation_tasks.vector_search_evaluation_task import (
    VectorSearchEvaluationTask,
)
from scripts.metrics.context_precision_metric import ContextPrecisionWrapper
from scripts.metrics.context_recall_metric import ContextRecallWrapper
from scripts.metrics.faithfulness_metric import FaithfulnessWrapper
from scripts.metrics.g_eval_metrics import GEvalWrapper

opik_tracer = OpikTracer(project_name=PROJECT_NAME)

litellm.drop_params = True


class ExperimentRunner:
    def __init__(
        self,
        client: Opik,
        dataset_name: str,
        project_name: str,
        dataset_path: Path,
        model_source: Literal["remote", "local"],
    ) -> None:
        self._client = client
        self._dataset = self._client.get_dataset(name=dataset_name)
        self.project_name = project_name
        self._model_source = model_source
        self._evaluation_criteria_map = self._load_evaluation_criteria(dataset_path)

    def _load_evaluation_criteria(self, dataset_path: Path) -> dict[str, list[str]]:
        with dataset_path.open() as f:
            dataset_rows: list[dict[str, Any]] = yaml.safe_load(f) or []

        criteria_map: dict[str, list[str]] = {}
        for row in dataset_rows:
            question = row.get("question")
            criteria = row.get("evaluation_criteria") or []
            if not question or not isinstance(criteria, list) or not criteria:
                continue
            criteria_map[str(question)] = [str(criterion) for criterion in criteria]

        if not criteria_map:
            raise ValueError(
                f"No evaluation criteria found in dataset file: {dataset_path}"
            )

        return criteria_map

    def configure_scoring_metrics(self) -> list[BaseMetric]:
        evaluator_llm = self._get_evaluator_model_wrapper()

        context_precision = LLMContextPrecisionWithReference(llm=evaluator_llm)
        context_recall = ContextRecall(llm=evaluator_llm)
        faithfulness = Faithfulness(llm=evaluator_llm)
        metrics = [
            ContextPrecisionWrapper(context_precision),
            ContextRecallWrapper(context_recall),
            FaithfulnessWrapper(faithfulness),
            GEvalWrapper(
                task_introduction="You evaluate ChatAGH RAG answers for university questions.",
                model=self._get_evaluator_model_name(),
                project_name=self.project_name,
                evaluation_criteria_map=self._evaluation_criteria_map,
            ),
        ]

        return metrics

    def run_experiment(
        self,
        evaluation_task: VectorSearchEvaluationTask,
    ) -> EvaluationResult:
        return evaluate(
            dataset=self._dataset,
            task=evaluation_task.run,
            scoring_metrics=self.configure_scoring_metrics(),
            experiment_config=evaluation_task.experiment_config,
            project_name=self.project_name,
            experiment_name=evaluation_task.task_name,
            task_threads=1,
        )

    def _get_evaluator_model_wrapper(self) -> BaseRagasLLM:
        if self._model_source == "remote":

            def build_remote_model() -> BaseChatModel:
                selected_model = draw_from_list(
                    candidates_with_weights=MODELS_WITH_RPM,
                    draw_counts=model_draw_counts,
                )
                api_key = draw_from_list(
                    candidates_with_weights={key: 1 for key in GEMINI_API_KEYS},
                    draw_counts=gemini_api_key_draw_counts,
                )
                return ChatGoogleGenerativeAI(
                    model=selected_model,
                    api_key=api_key,
                    temperature=1.0,
                    callbacks=[opik_tracer],
                )

            return ResilientLangchainLLMWrapper(
                model_factory=build_remote_model,
                max_attempts=len(MODELS_WITH_RPM),
            )

        model: BaseChatModel = ChatOllama(
            model=OLLAMA_MODEL,
            temperature=0.7,
            callbacks=[opik_tracer],
        )

        langchain_model = cast(BaseLanguageModel, model)
        return LangchainLLMWrapper(cast(Any, langchain_model))

    def _get_evaluator_model_name(self) -> str:
        if self._model_source == "remote":
            return draw_from_list(
                candidates_with_weights=MODELS_WITH_RPM,
                draw_counts=model_draw_counts,
            )
        else:
            return f"ollama/{OLLAMA_MODEL}"


T = TypeVar("T")


class ResilientLangchainLLMWrapper(LangchainLLMWrapper):
    """Langchain wrapper that retries with a new Gemini model on rate limits."""

    def __init__(
        self,
        model_factory: Callable[[], BaseChatModel],
        max_attempts: int,
    ) -> None:
        self._model_factory = model_factory
        self._max_attempts = max(1, max_attempts)
        initial_model = cast(BaseLanguageModel, self._model_factory())
        super().__init__(initial_model)

    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float | None = None,
        stop: list[str] | None = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        parent_generate = super().generate_text
        return self._call_with_fallback(
            lambda: parent_generate(prompt, n, temperature, stop, callbacks)
        )

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float | None = None,
        stop: list[str] | None = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        parent_agenerate = super().agenerate_text
        return await self._call_with_fallback_async(
            lambda: parent_agenerate(prompt, n, temperature, stop, callbacks)
        )

    def _swap_model(self) -> None:
        new_model = cast(BaseLanguageModel, self._model_factory())
        self.langchain_llm = new_model

    def _call_with_fallback(self, fn: Callable[[], T]) -> T:
        last_exc: ResourceExhausted | None = None
        for idx in range(self._max_attempts):
            try:
                return fn()
            except ResourceExhausted as exc:
                last_exc = exc
                self._handle_rate_limit(exc, idx, self._max_attempts)
                if idx == self._max_attempts - 1:
                    break
                self._swap_model()
        assert last_exc is not None
        raise last_exc

    async def _call_with_fallback_async(self, fn: Callable[[], Awaitable[T]]) -> T:
        last_exc: ResourceExhausted | None = None
        for idx in range(self._max_attempts):
            try:
                return await fn()
            except ResourceExhausted as exc:  # pragma: no cover - API branch
                last_exc = exc
                self._handle_rate_limit(exc, idx, self._max_attempts)
                if idx == self._max_attempts - 1:
                    break
                self._swap_model()
        assert last_exc is not None
        raise last_exc

    def _handle_rate_limit(
        self, exc: ResourceExhausted, attempt_idx: int, total_attempts: int
    ) -> None:
        current_model = getattr(self.langchain_llm, "model", "unknown")
        logger.warning(
            "Evaluator Gemini model %s rate limited (attempt %s/%s): %s",
            current_model,
            attempt_idx + 1,
            total_attempts,
            exc,
        )
