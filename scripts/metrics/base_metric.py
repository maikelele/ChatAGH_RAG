from opik import track
from opik.evaluation.metrics import BaseMetric
from opik.evaluation.metrics.score_result import ScoreResult
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import SingleTurnMetric


class BaseMetricWrapper(BaseMetric):
    def __init__(self, metric: SingleTurnMetric, name: str = "base_metric"):
        self.metric = metric
        self.name = name

    @track(name="base_score()")
    def base_score(self, row) -> ScoreResult:
        row = SingleTurnSample(**row)
        score_result = self.metric.single_turn_score(row)

        return ScoreResult(value=score_result, name=self.name)
