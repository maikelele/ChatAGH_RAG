from typing import Any, Dict, List

from opik.evaluation.metrics import BaseMetric, GEval
from opik.evaluation.metrics.score_result import ScoreResult


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

        metric = GEval(
            task_introduction=self._task_introduction,
            evaluation_criteria="\n".join(f"- {criterion}" for criterion in criteria),
            model=self._model,
            project_name=self._project_name,
        )
        self._metric_cache[question] = metric
        return metric
