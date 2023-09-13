from typing import Any, Dict

import determined as det
import torch
from sentence_transformers import SentenceTransformer

from mteb import MTEB

device = "cuda" if torch.cuda.is_available() else "cpu"


def main(hparams: Dict[str, Any], core_context: det.core.Context) -> None:
    # Define the sentence-transformers model name
    model_name = hparams["model_name"]

    model = SentenceTransformer(model_name, device=device)
    evaluation = MTEB(
        tasks=hparams.get("tasks"),
        task_types=hparams.get("task_types"),
        task_categories=hparams.get("task_categories"),
        task_langs=hparams.get("task_langs"),
    )
    results = evaluation.run(model, output_folder=f"results/{ model_name }")
    metrics: Dict[str, Any] = {}
    for task, res_dict in results.items():
        for res_name, res_val in res_dict.items():
            metrics[task + "_" + res_name] = res_val

    if core_context.distributed.rank == 0:
        core_context.train.report_validation_metrics(steps_completed=1, metrics=metrics)


if __name__ == "__main__":
    info = det.get_cluster_info()
    assert info, "This script must run on a determined cluster."
    hparams = info.trial.hparams
    distributed = det.core.DistributedContext.from_torch_distributed()
    with det.core.init(distributed=distributed) as core_context:
        main(hparams, core_context)
