from pathlib import Path
from typing import Any, Protocol

import litellm
import yaml  # type: ignore[import-untyped]
from langchain_community.chat_models import ChatOllama
from opik import Opik, Prompt, evaluate
from opik.evaluation.evaluation_result import EvaluationResult
from opik.evaluation.metrics import BaseMetric
from opik.integrations.langchain import OpikTracer
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    ContextRecall,
    Faithfulness,
    LLMContextPrecisionWithReference,
)

from scripts.consts import PROJECT_NAME
from scripts.metrics.context_precision_metric import ContextPrecisionWrapper
from scripts.metrics.context_recall_metric import ContextRecallWrapper
from scripts.metrics.faithfulness_metric import FaithfulnessWrapper
from scripts.metrics.g_eval_metrics import GEvalWrapper

opik_tracer = OpikTracer(project_name=PROJECT_NAME)

litellm.drop_params = True


class EvaluationTaskProtocol(Protocol):
    task_name: str
    experiment_config: dict[str, Any]
    prompts: list[Prompt]

    def run(self, input_data: dict[str, str]) -> dict[str, Any]: ...


class ExperimentRunner:
    def __init__(
        self,
        client: Opik,
        dataset_name: str,
        project_name: str,
        criteria_evaluator_model_name: str,
        ragas_evaluator_model_name: str,
        evaluator_embeddings_model_name: str,
        dataset_path: Path,
    ) -> None:
        self._client = client
        self._dataset = self._client.get_dataset(name=dataset_name)
        self.project_name = project_name
        self.criteria_evaluator_model_name = criteria_evaluator_model_name
        self.ragas_evaluator_model_name = ragas_evaluator_model_name
        self.evaluator_embeddings_model_name = evaluator_embeddings_model_name
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
        evaluator_llm = LangchainLLMWrapper(
            ChatOllama(model="gemma3", temperature=0.7, callbacks=[opik_tracer])
        )

        context_precision = LLMContextPrecisionWithReference(llm=evaluator_llm)
        context_recall = ContextRecall(llm=evaluator_llm)
        faithfulness = Faithfulness(llm=evaluator_llm)
        metrics = [
            ContextPrecisionWrapper(context_precision),
            ContextRecallWrapper(context_recall),
            FaithfulnessWrapper(faithfulness),
            GEvalWrapper(
                task_introduction="You evaluate ChatAGH RAG answers for university questions.",
                model="ollama/gemma3",
                project_name=self.project_name,
                evaluation_criteria_map=self._evaluation_criteria_map,
            ),
        ]

        return metrics

    def run_experiment(
        self, evaluation_task: EvaluationTaskProtocol
    ) -> EvaluationResult:
        return evaluate(
            dataset=self._dataset,
            task=evaluation_task.run,
            scoring_metrics=self.configure_scoring_metrics(),
            experiment_config=evaluation_task.experiment_config,
            project_name=self.project_name,
            experiment_name=evaluation_task.task_name,
            prompts=evaluation_task.prompts,
            task_threads=1,
        )
