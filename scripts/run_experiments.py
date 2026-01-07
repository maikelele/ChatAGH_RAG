from hashlib import shake_256
from pathlib import Path
from typing import Literal

import opik
import pandas as pd
import yaml  # type: ignore[import-untyped, unused-ignore]
from dotenv import load_dotenv
from opik import Opik
from opik.rest_api.core import ApiError

from chat_agh.utils.singletons import logger
from scripts.consts import (
    DATASET_NAME,
    PROJECT_NAME,
)
from scripts.evaluation_tasks.base import SearchParametersOverride
from scripts.evaluation_tasks.vector_search_evaluation_task import (
    VectorSearchEvaluationTask,
)
from scripts.experiment_runner import ExperimentRunner

opik.configure(url="http://localhost:5173", use_local=True)


def _content_hash(path: Path) -> str:
    length = 8
    with path.open("rb") as f:
        data = f.read()
    return shake_256(data).hexdigest(length // 2)


def _yaml_to_df(path: Path) -> pd.DataFrame:
    logger.info(f"Processing {path}")
    with path.open() as f:
        data = yaml.safe_load(f)
    return pd.DataFrame(data)


def _create_opik_dataset(client: Opik, dataset_name: str, dataset_path: Path) -> None:
    df = _yaml_to_df(dataset_path)
    logger.info(f"Inserting {dataset_name} to Opik")
    try:
        dataset = client.create_dataset(name=dataset_name)
        dataset.insert_from_pandas(dataframe=df)
    except ApiError:
        logger.info(f"{dataset_name} already uploaded to Opik")


def _prepare_opik_dataset(client: Opik, dataset_name: str) -> str:
    dataset_path = Path(__file__).parent / "datasets" / (dataset_name + ".yaml")

    dataset_id = _content_hash(dataset_path)

    opik_dataset_name = f"{dataset_name}_{dataset_id}"

    _create_opik_dataset(client, opik_dataset_name, dataset_path)

    return opik_dataset_name


def get_model_source() -> Literal["remote", "local"]:
    model_source: str = input("Model source (r - remote / l - local): ")
    if not model_source:
        raise ValueError("model_source must not be empty")
    return normalize(model_source)


def normalize(model_source: str) -> Literal["remote", "local"]:
    if model_source.lower()[0] == "r":
        return "remote"
    elif model_source.lower()[0] == "l":
        return "local"
    else:
        raise ValueError("model_source can be either 'remote' or 'local'")


if __name__ == "__main__":
    load_dotenv()

    client = Opik(project_name="chat_agh_rag")
    dataset_name = _prepare_opik_dataset(client, DATASET_NAME)
    dataset_file_path = Path(__file__).parent / "datasets" / f"{DATASET_NAME}.yaml"

    model_source = get_model_source()

    runner = ExperimentRunner(
        client=client,
        dataset_name=dataset_name,
        project_name=PROJECT_NAME,
        model_source=model_source,
        dataset_path=dataset_file_path,
    )

    vector_search_settings: list[SearchParametersOverride] = [
        SearchParametersOverride(
            mode="lexical",
            fuzzy=False,
            lexical_limit=10,
        ),
        SearchParametersOverride(
            mode="lexical",
            fuzzy=False,
            lexical_limit=15,
        ),
    ]

    tasks = [
        VectorSearchEvaluationTask(
            collection_names=[
                "cluster_0",
                "cluster_6",
                "cluster_7",
                "cluster_8",
                "cluster_9",
            ],
            search_setting=setting,
        )
        for setting in vector_search_settings
    ]

    for task in tasks:
        logger.info(f"Running experiment: {task._search_setting.to_dict()}")
        runner.run_experiment(task)
