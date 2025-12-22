from typing import Any

from opik.evaluation.metrics.score_result import ScoreResult
from ragas.metrics import ContextRecall

from scripts.metrics.base_metric import BaseMetricWrapper


class ContextRecallWrapper(BaseMetricWrapper):
    def __init__(self, metric: ContextRecall) -> None:
        super().__init__(metric=metric, name="context_recall_metric")

    def score(
        self,
        question: str,
        answer: str,
        retrieved_contexts: list[str],
        response: str,
        **_: dict[str, Any],
    ) -> ScoreResult:
        row = {
            "user_input": question,
            "reference": answer,
            "retrieved_contexts": retrieved_contexts,
            "response": response,
        }
        return self.base_score(row)
