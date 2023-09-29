import logging
from typing import Any, Dict, List

import determined as det
import torch
import torch.nn as nn
from InstructorEmbedding import INSTRUCTOR
from sentence_transformers import SentenceTransformer

from mteb import MTEB

device = "cuda" if torch.cuda.is_available() else "cpu"


class TESTModel(nn.Module):
    """
    Trivial test model implementing the minimal interface described in the README.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)
        if not hasattr(self, "device"):
            self.device = "cuda"

    def encode(self, sentences, batch_size: int = 32, **kwargs) -> List[torch.Tensor]:
        return torch.randn(len(sentences), 128, device=self.device)


MODEL_TYPE_DICT = {
    "INSTRUCTOR": INSTRUCTOR,
    "SentenceTransformer": SentenceTransformer,
    "TESTModel": TESTModel,
}


def main(hparams: Dict[str, Any], core_context: det.core.Context) -> None:
    # Define the sentence-transformers model name
    model_name = hparams["model_name"]
    model_type = hparams["model_type"]
    assert (
        model_type in MODEL_TYPE_DICT
    ), f"Expected model_type to be in {list(MODEL_TYPE_DICT)} received {model_type}"

    if hparams.get("load_from_uuid"):
        with core_context.checkpoint.restore_path(hparams["load_from_uuid"]) as path:
            logging.info(f"Loading model from checkpoint {hparams['load_from_uuid']}")
            model = MODEL_TYPE_DICT[model_type](path, device=device)
            model_name += f"_{hparams['load_from_uuid']}"
    else:
        model = MODEL_TYPE_DICT[model_type](model_name, device=device)
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
