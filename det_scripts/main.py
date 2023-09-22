import logging
import os
from typing import Any, Dict

import determined as det
import torch
from InstructorEmbedding import INSTRUCTOR
from sentence_transformers import SentenceTransformer

from mteb import MTEB

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_TYPE_DICT = {"INSTRUCTOR": INSTRUCTOR, "SentenceTransformer": SentenceTransformer}


def main(hparams: Dict[str, Any], core_context: det.core.Context) -> None:
    # Define the sentence-transformers model name
    model_name = hparams["model_name"]
    model_type = hparams["model_type"]
    assert (
        model_type in MODEL_TYPE_DICT
    ), f"Expected model_type to be in {list(MODEL_TYPE_DICT)} received {model_type}"

    model = MODEL_TYPE_DICT[model_type](model_name, device=device)
    assert hasattr(model, "start_multi_process_pool"), "No start_multi_process_pool attr"
    evaluation = MTEB(
        tasks=hparams.get("tasks"),
        task_types=hparams.get("task_types"),
        task_categories=hparams.get("task_categories"),
        task_langs=hparams.get("task_langs"),
        hparams=hparams,
    )
    results = evaluation.run(
        model,
        output_folder=f"results/{ model_name }",
        core_context=core_context,
        model_name=model_name,
        hparams=hparams,
        **hparams.get("evaluate_kwargs", {}),
    )
    logging.info(f"Results: {results}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=det.LOG_FORMAT)
    info = det.get_cluster_info()
    assert info, "This script must run on a determined cluster."
    hparams = info.trial.hparams
    try:
        distributed = det.core.DistributedContext.from_torch_distributed()
    except KeyError:
        distributed = None
    with det.core.init(distributed=distributed) as core_context:
        main(hparams, core_context)
