import copy
import itertools
import logging
import os
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, Dict, List, Optional

import datasets
import determined as det
import numpy as np
import omegaconf
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import transformers
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import mteb

WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
RANK = int(os.getenv("RANK", 0))
WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
USE_WANDB = WANDB_API_KEY is not None
DEVICE = f"cuda:{RANK}"
DEBUG_TASK = "zeroshot"
DEBUG_STEPS = 10

"""
Script for training decoder-only models on the medi dataset introduced in the INSTRUCTOR paper
https://instructor-embedding.github.io

GG_NOTE: Some HF models raise errors in distributed training due to the below in-place operation:
https://github.com/huggingface/trl/issues/274
"""


class TESTModel(nn.Module):
    """
    Trivial test model.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)
        if not hasattr(self, "device"):
            self.device = DEVICE

    def forward(self, *args, **kwargs):
        return torch.randn(kwargs["input_ids"].shape[0], 128, device=self.device)

    def encode(self, sentences, batch_size: int = 32, **kwargs) -> List[torch.Tensor]:
        return torch.randn(len(sentences), 128, device=self.device)


def build_model_and_optimizer(hparams: omegaconf.OmegaConf, core_context: det.core.Context):
    model = mteb.models.DetInstructor(hparams, core_context)
    model.to(DEVICE)
    if WORLD_SIZE > 1:
        model = DDP(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hparams.lr)
    if hparams.get("load_from_uuid"):
        with core_context.checkpoint.restore_path(hparams.load_from_uuid) as path:
            logging.info(f"Loading optimizer state from {hparams.load_from_uuid}")
            optimizer.load_state_dict(torch.load(path.joinpath("optim.pt")))
    return model, optimizer


def build_train_val_dataset_dicts(hparams: omegaconf.OmegaConf):
    train_dataset_dict = datasets.load_dataset(hparams.train_dataset)
    val_dataset_dict = datasets.load_dataset(hparams.val_dataset)
    return train_dataset_dict, val_dataset_dict


def build_train_val_dataloaders_dict(
    tokenizer,
    train_dataset_dict: datasets.DatasetDict,
    val_dataset_dict: datasets.DatasetDict,
    hparams: omegaconf.OmegaConf,
):
    def collator(examples):
        keys = ("pos", "neg", "query")
        examples_dict = defaultdict(list)
        for example in examples:
            for k in keys:
                # GG_NOTE: I would have thought that we should `join` the sentences in example[k]
                # but in the INSTRUCTOR MTEB benchmarking code, they instead tokenize List[List[str]
                # inputs.
                examples_dict[k].append(example[k])
        inputs_dict = {
            key: tokenizer(
                examples_dict[key],
                return_tensors="pt",
                padding=True,
                return_attention_mask=True,
                truncation=True,
            )
            for key in examples_dict
        }
        return inputs_dict

    train_dataloader_dict = {}
    val_dataloader_dict = {}
    for task_name in train_dataset_dict:
        train_dset = train_dataset_dict[task_name]
        val_dset = val_dataset_dict[task_name]
        # GG_TODO: Shuffling
        if WORLD_SIZE == 1:
            train_sampler = torch.utils.data.SequentialSampler(train_dset)
            val_sampler = torch.utils.data.SequentialSampler(val_dset)
        else:
            train_sampler = torch.utils.data.DistributedSampler(train_dset)
            val_sampler = torch.utils.data.DistributedSampler(val_dset)
        train_dataloader_dict[task_name] = torch.utils.data.DataLoader(
            train_dset,
            batch_size=hparams.batch_size,
            collate_fn=collator,
            sampler=train_sampler,
            drop_last=True,
        )
        val_dataloader_dict[task_name] = torch.utils.data.DataLoader(
            val_dset,
            batch_size=hparams.get("val_batch_size") or hparams.batch_size,
            collate_fn=collator,
            sampler=val_sampler,
            drop_last=True,
        )
    return train_dataloader_dict, val_dataloader_dict


def save_model():
    pass


def process_batch(model, inputs, hparams, tokenizer) -> Dict[str, Any]:
    embeds = model(inputs)
    batch_size, _ = embeds["query"].shape

    all_scores = None

    # GG_NOTE: Compute the cos-based loss over the scores between the query and its corresponding
    # positive entry, competing against the score from the query against all negative entries
    # in the batch.
    def dot(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
        """Dot products of 1D or 2D tensors with the same semantics as nn.CosineSimilarity."""
        assert tensor1.shape == tensor2.shape
        if tensor1.dim() == 1:
            return tensor1 @ tensor2
        if tensor1.dim() == 2:
            assert tensor1.shape[0] == 1
            return (tensor1 * tensor2).sum(dim=-1)
        raise ValueError(f"Received unexpected {tensor1.dim()} dimensional tensor. Expected 1D/2D.")

    similarity_fn_dict = {"cos": nn.CosineSimilarity(dim=-1), "dot": dot}

    similarity_fn = similarity_fn_dict[hparams.similarity_fn]
    for batch_idx in range(batch_size):
        query_emb, pos_emb = embeds["query"][batch_idx], embeds["pos"][batch_idx]
        # GG_NOTE: compute the cos between the query and the positive result.
        # Make the result 2D so we can concatenate across batch and other-sample dims.
        cur_score = similarity_fn(query_emb, pos_emb)[None, None] / hparams.temp
        # GG_NOTE: compute the cos between the query and all negative results and stack
        # along last dimension
        for neg_emb in embeds["neg"]:
            one_neg_score = similarity_fn(query_emb, neg_emb)[None, None] / hparams.temp
            cur_score = torch.cat([cur_score, one_neg_score], dim=-1)

        # And also add all positive examples from other elements in the batch
        for other_batch_idx in range(batch_size):
            if other_batch_idx == batch_idx:
                continue
            other_pos_emb = embeds["pos"][other_batch_idx]
            one_neg_score = similarity_fn(pos_emb, other_pos_emb)[None, None] / hparams.temp
            cur_score = torch.cat([cur_score, one_neg_score], dim=-1)

        # And finally stack these results along the zeroth dim.
        if all_scores is None:
            all_scores = cur_score
        else:
            all_scores = torch.cat([all_scores, cur_score], dim=0)
    assert all_scores is not None
    labels = torch.zeros(all_scores.size(0)).long().to(DEVICE)
    loss = F.cross_entropy(all_scores, labels)
    if hparams.similarity_fn == "cos":
        with torch.no_grad():
            # In the cosine strategy, the minimum possible loss is not zero, when defined as above.
            # For convenience subtract a constant to make the minimum zero.
            perfect_pred = -1.0 * torch.ones_like(all_scores) / hparams.temp
            perfect_pred[:, 0] = 1.0 / hparams.temp
            perfect_loss = F.cross_entropy(perfect_pred, labels)
        loss = loss - perfect_loss.item()

    # GG_NOTE: Also use the "bi-directional in-batch" loss of Ni et al. 2021 2112.07899
    # where they also compute losses for "document-to-question matching".
    all_bidirectional_scores = None
    for batch_idx in range(batch_size):
        pos_emb, query_emb = embeds["pos"][batch_idx], embeds["query"][batch_idx]
        cur_score = similarity_fn(pos_emb, query_emb)[None, None] / hparams.temp

        for other_batch_idx in range(batch_size):
            if other_batch_idx == batch_idx:
                continue
            other_query_emb = embeds["query"][other_batch_idx]
            one_neg_score = similarity_fn(pos_emb, other_query_emb)[None, None] / hparams.temp
            cur_score = torch.cat([cur_score, one_neg_score], dim=-1)
        if all_bidirectional_scores is None:
            all_bidirectional_scores = cur_score
        else:
            all_bidirectional_scores = torch.cat([all_bidirectional_scores, cur_score], dim=0)
    assert all_bidirectional_scores is not None
    labels_bidirectional = torch.zeros(all_bidirectional_scores.size(0)).long().to(DEVICE)
    loss = loss + F.cross_entropy(all_bidirectional_scores, labels_bidirectional)
    if hparams.similarity_fn == "cos":
        with torch.no_grad():
            # In the cosine strategy, the minimum possible loss is not zero, when defined as above.
            # For convenience subtract a (batch-size-dependent) constant to make the minimum zero.
            perfect_pred_bidirectional = (
                -1.0 * torch.ones_like(all_bidirectional_scores) / hparams.temp
            )
            perfect_pred_bidirectional[:, 0] = 1.0 / hparams.temp
            perfect_loss_bidirectional = F.cross_entropy(
                perfect_pred_bidirectional, labels_bidirectional
            )
        loss = loss - perfect_loss_bidirectional.item()

    metrics = {"loss": loss}
    return metrics


def train_one_step(model, tokenizer, train_data_iter, optimizer, scaler=None) -> Dict[str, Any]:
    model.train()
    inputs = {k: v.to(DEVICE) for k, v in next(train_data_iter).items()}
    if hparams.get("debug"):
        shapes = {k: {kk: t.shape for kk, t in v.items()} for k, v in inputs.items()}
        logging.info(f"infnite input shapes: {shapes}")
    train_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16) if hparams.amp else nullcontext()
    )
    with train_context:
        metrics = process_batch(model=model, inputs=inputs, hparams=hparams, tokenizer=tokenizer)
        loss = metrics["loss"]
    if hparams.amp:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
    return metrics


def build_infinite_data_iterator(dataloader_dict, hparams):
    """Randomly selects a category and returns the next item from the relevant dataloader."""
    num_samples = sum(len(v) for v in dataloader_dict.values())
    train_probs = {k: len(v) / num_samples for k, v in dataloader_dict.items()}
    task_list, prob_list = [], []
    for k, p in train_probs.items():
        task_list.append(k)
        prob_list.append(p)
    # GG_TODO: Restore random state after restart
    rng = np.random.default_rng(hparams.seed)
    iter_dict = {k: iter(v) for k, v in dataloader_dict.items()}

    # Use a single small sub-dataset when debugging
    while True:
        next_cat = rng.choice(task_list, p=prob_list)
        data_iter = iter_dict[next_cat]

        try:
            yield next(data_iter)
        except StopIteration:
            iter_dict[next_cat] = iter(dataloader_dict[next_cat])
            data_iter = iter_dict[next_cat]
            yield next(data_iter)


def validate(model, tokenizer, val_data_iter, hparams) -> Optional[Dict[str, Any]]:
    model.eval()
    with torch.no_grad():
        all_val_metrics = defaultdict(list)
        max_val_steps = hparams.val_steps if not hparams.debug else DEBUG_STEPS
        for val_step in tqdm(range(max_val_steps), desc="Validating"):
            inputs = next(val_data_iter)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            metrics = process_batch(model, inputs, hparams, tokenizer)
            for k, v in metrics.items():
                all_val_metrics[k].append(v.item())
        mean_val_metrics = {k: torch.tensor(v).mean() for k, v in all_val_metrics.items()}
        if WORLD_SIZE > 1:
            for v in mean_val_metrics.values():
                dist.reduce(v.to(DEVICE), dst=0, op=dist.ReduceOp.SUM)
        if RANK == 0:
            return {"val_" + k: v.item() / WORLD_SIZE for k, v in mean_val_metrics.items()}
        else:
            return None


def reduce_and_report_train_metrics(
    train_metrics_collected: Dict[str, List[float]], step: int
) -> None:
    mean_train_metrics = {k: torch.tensor(v).mean() for k, v in train_metrics_collected.items()}
    if WORLD_SIZE > 1:
        for v in mean_train_metrics.values():
            dist.reduce(v.to(DEVICE), dst=0, op=dist.ReduceOp.SUM)
    if RANK == 0:
        reduced_train_metrics = {k: v.item() / WORLD_SIZE for k, v in mean_train_metrics.items()}
        if USE_WANDB:
            wandb.log(
                reduced_train_metrics,
                step=step,
            )
        core_context.train.report_training_metrics(
            steps_completed=step, metrics=reduced_train_metrics
        )


def save_model_and_optimizer(model, optimizer, core_context: det.core.Context, step: int) -> None:
    metadata = {"steps_completed": step}
    with core_context.checkpoint.store_path(metadata) as (path, storage_id):
        torch.save(optimizer.state_dict(), path.joinpath("optim.pt"))
        model.save_pretrained(path)


def main(hparams: omegaconf.OmegaConf, core_context: det.core.Context):
    if hparams.debug:
        torch.autograd.set_detect_anomaly(True)

    scaler = torch.cuda.amp.GradScaler() if hparams.amp else None

    transformers.set_seed(hparams.seed)

    model, optimizer = build_model_and_optimizer(hparams, core_context)

    train_dataset_dict, val_dataset_dict = build_train_val_dataset_dicts(hparams)
    train_dataloader_dict, val_dataloader_dict = build_train_val_dataloaders_dict(
        model.hf_tokenizer, train_dataset_dict, val_dataset_dict, hparams
    )

    train_metrics_collected = defaultdict(list)
    step = 0

    train_data_iter = build_infinite_data_iterator(train_dataloader_dict, hparams)
    val_data_iter = build_infinite_data_iterator(val_dataloader_dict, hparams)
    max_steps = DEBUG_STEPS if hparams.debug else hparams.steps
    for step in tqdm(range(1, max_steps + 1), desc="Training"):
        train_step_metrics = train_one_step(
            model=model,
            tokenizer=model.hf_tokenizer,
            train_data_iter=train_data_iter,
            optimizer=optimizer,
            scaler=scaler,
        )
        for k, v in train_step_metrics.items():
            train_metrics_collected[k].append(v.item())
        optimizer.zero_grad(set_to_none=True)

        report_rate = DEBUG_STEPS if hparams.debug else hparams.report_rate
        if step % report_rate == 0:
            reduce_and_report_train_metrics(
                train_metrics_collected=train_metrics_collected, step=step
            )
            train_metrics_collected = defaultdict(list)

        val_rate = DEBUG_STEPS if hparams.debug else hparams.val_rate
        if step % val_rate == 0:
            if RANK == 0:
                save_model_and_optimizer(model, optimizer, core_context, step)
            # Only RANK == 0 will get the fully reduced metrics, others get None.
            val_metrics = validate(
                model=model,
                tokenizer=model.hf_tokenizer,
                val_data_iter=val_data_iter,
                hparams=hparams,
            )
            if RANK == 0:
                if USE_WANDB:
                    wandb.log(val_metrics, step=step)
                core_context.train.report_validation_metrics(
                    steps_completed=step, metrics=val_metrics
                )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=det.LOG_FORMAT)
    info = det.get_cluster_info()
    assert info, "This script must run on a determined cluster."

    hparams = omegaconf.OmegaConf.create(info.trial.hparams)
    try:
        distributed = det.core.DistributedContext.from_torch_distributed()
        torch.distributed.init_process_group("nccl")
    except KeyError:
        distributed = None

    with logging_redirect_tqdm():
        with det.core.init(distributed=distributed) as core_context:
            if USE_WANDB and core_context.distributed.rank == 0:
                logging.info("Reporting results to wandb ... ")
                assert isinstance(hparams.model_name, str)
                name = (
                    hparams.model_name + f"_bsz_{hparams.batch_size}"
                    f"_lr_{hparams.lr}"
                    + f"_strat_{hparams.embed_strat}"
                    + f"_temp_{hparams.temp}"
                    + f"_{hparams.similarity_fn}"
                )
                config = copy.deepcopy(dict(hparams))
                config["exp_id"] = info.trial.experiment_id
                config["trial_id"] = info.trial.trial_id
                config["world_size"] = WORLD_SIZE
                run = wandb.init(
                    project="Embeddings",
                    config=config,
                    name=name,
                    job_type="medi_train",
                    tags=["train"],
                )
            main(hparams, core_context)
