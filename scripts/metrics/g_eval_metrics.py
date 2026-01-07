from typing import Any, Dict, List, Optional

from opik.evaluation.metrics import BaseMetric, GEval
from opik.evaluation.metrics.score_result import ScoreResult
from opik.evaluation.models.litellm.litellm_chat_model import LiteLLMChatModel

from chat_agh.utils.utils import (
    GEMINI_API_KEYS,
    draw_from_list,
    gemini_api_key_draw_counts,
)


class GEvalWrapper(BaseMetric):
    """Adapter around Opik's GEval metric with per-question criteria."""

    def __init__(
        self,
        *,
        task_introduction: str,
        model: str,
        project_name: str,
        evaluation_criteria_map: Dict[str, List[str]],
    ) -> None:
        self._task_introduction = task_introduction
        self._model = model
        self._project_name = project_name
        self._evaluation_criteria_map = evaluation_criteria_map
        self._metric_cache: Dict[str, GEval] = {}
        self.name = "geval"

    def score(
        self,
        question: str,
        answer: str,
        retrieved_contexts: list[str],
        response: str,
        **_: dict[str, Any],
    ) -> ScoreResult:
        del answer, retrieved_contexts

        metric = self._get_metric(question)
        return metric.score(output=response)

    def _get_metric(self, question: str) -> GEval:
        metric = self._metric_cache.get(question)
        if metric:
            return metric

        criteria = self._evaluation_criteria_map.get(question)
        if not criteria:
            raise ValueError(
                "No evaluation criteria found for question. "
                "Ensure dataset entries include evaluation_criteria."
            )

        metric_model = self._build_metric_model()

        metric = GEval(
            task_introduction=self._task_introduction,
            evaluation_criteria="\n".join(f"- {criterion}" for criterion in criteria),
            model=metric_model or self._model,
            project_name=self._project_name,
        )
        self._metric_cache[question] = metric
        return metric

    def _build_metric_model(self) -> Optional[LiteLLMChatModel]:
        """Return a LiteLLM model with API key support when using Gemini."""

        if not self._model.startswith("gemini-"):
            return None

        gemini_model_name = self._model
        if not gemini_model_name.startswith("gemini/"):
            gemini_model_name = f"gemini/{gemini_model_name}"

        return LiteLLMChatModel(
            model_name=gemini_model_name,
            api_key=draw_from_list(
                candidates_with_weights={key: 1 for key in GEMINI_API_KEYS},
                draw_counts=gemini_api_key_draw_counts,
            ),
        )
