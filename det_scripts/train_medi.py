import logging
import os
from collections import defaultdict
from typing import Any, Dict, List

import datasets
import determined as det
import numpy as np
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP

WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
RANK = int(os.getenv("RANK", 0))
WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
USE_WANDB = WANDB_API_KEY is not None


class TESTModel(nn.Module):
    """
    Trivial test model.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)
        if not hasattr(self, "device"):
            self.device = "cuda"

    def forward(self, *args, **kwargs):
        return torch.randn(kwargs["input_ids"].shape[0], 128, device=self.device)

    def encode(self, sentences, batch_size: int = 32, **kwargs) -> List[torch.Tensor]:
        return torch.randn(len(sentences), 128, device=self.device)


def build_model_and_optimizer(hparams: omegaconf.OmegaConf):
    model = transformers.AutoModel.from_pretrained(hparams.model_name)
    if WORLD_SIZE > 1:
        model = DDP(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hparams.lr)
    return model, optimizer


def build_train_val_dataset_dicts(hparams: omegaconf.OmegaConf):
    train_dataset_dict = datasets.load_dataset(hparams.train_dataset)
    val_dataset_dict = datasets.load_dataset(hparams.val_dataset)
    return train_dataset_dict, val_dataset_dict


def build_train_val_dataloaders_dict(
    train_dataset_dict: datasets.DatasetDict,
    val_dataset_dict: datasets.DatasetDict,
    hparams: omegaconf.OmegaConf,
):
    tokenizer_name = hparams.get("tokenizer_name") or hparams.model_name
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.paddding_side = "right"

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
                examples_dict[key], return_tensors="pt", padding=True, return_attention_mask=True
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
            batch_size=hparams.batch_size,
            collate_fn=collator,
            sampler=val_sampler,
            drop_last=True,
        )
    return train_dataloader_dict, val_dataloader_dict


def save_model():
    pass


def process_batch(model, inputs, hparams) -> Dict[str, Any]:
    embeds = {}
    for k in ["query", "pos", "neg"]:
        raw_model_outputs = model(**inputs[k])
        if hparams.embed_strat == "last":
            # Need to take the final state according to the attention masks
            attn_mask_sum = inputs[k]["attention_mask"].sum(dim=-1) - 1
            embeds[k] = raw_model_outputs.last_hidden_state[
                range(raw_model_outputs.last_hidden_state.shape[0]), attn_mask_sum
            ]
        elif hparams.embed_strat == "weighted":
            _, seq_len, _ = raw_model_outputs.last_hidden_state.shape
            # Create the weights
            linear_weigths = torch.arange(
                start=1,
                end=seq_len + 1,
                dtype=raw_model_outputs.last_hidden_state.dtype,
                device=raw_model_outputs.last_hidden_state.device,
            ).flip(dims=(0,))[None, ..., None]
            # Each weight needs to be weights by different values
            weights /= weights.sum(dim=1)
            logging.info("Using weights {weights}")
            embeds[k] = raw_model_outputs.last_hidden_state * weights
        else:
            raise ValueError(f"Recieved unexpected embed_strat: {hparams.embed_strat}")

    batch_size, _ = embeds["query"].shape
    device = embeds["query"].device

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
        cur_score = similarity_fn(query_emb, pos_emb)[None, None] / hparams.cl_temperature
        # GG_NOTE: compute the cos between the query and all negative results and stack
        # along last dimension
        for neg_emb in embeds["neg"]:
            one_neg_score = similarity_fn(query_emb, neg_emb)[None, None] / hparams.cl_temperature
            cur_score = torch.cat([cur_score, one_neg_score], dim=-1)

        # And also add all positive examples from other elements in the batch
        for other_batch_idx in range(batch_size):
            if other_batch_idx == batch_idx:
                continue
            other_pos_emb = embeds["pos"][other_batch_idx]
            one_neg_score = (
                similarity_fn(pos_emb, other_pos_emb)[None, None] / hparams.cl_temperature
            )
            cur_score = torch.cat([cur_score, one_neg_score], dim=-1)

        # And finally stack these results along the zeroth dim.
        if all_scores is None:
            all_scores = cur_score
        else:
            all_scores = torch.cat([all_scores, cur_score], dim=0)
    labels = torch.zeros(all_scores.size(0)).long().to(device)
    loss = F.cross_entropy(all_scores, labels)
    if hparams.similarity_fn == "cos":
        with torch.no_grad():
            # In the cosine strategy, the minimum possible loss is not zero, when defined as above.
            # For convenience subtract a (batch-size-dependent) constant to make the minimum zero.
            num_cat_all_scores = all_scores.shape[-1]
            elems = torch.arange(num_cat_all_scores).to(device)
            perfect_pred = torch.where(
                elems > 0, -1.0 / hparams.cl_temperature, 1.0 / hparams.cl_temperature
            )
            perfect_loss = F.cross_entropy(perfect_pred[None], torch.zeros(1, device=device).long())
        loss -= perfect_loss.item()

    # GG_NOTE: Also use the "bi-directional in-batch" loss of Ni et al. 2021 2112.07899
    # where they also compute losses for "document-to-question matching".
    all_another_scores = None
    for batch_idx in range(batch_size):
        pos_emb, query_emb = embeds["pos"][batch_idx], embeds["query"][batch_idx]
        cur_score = similarity_fn(pos_emb, query_emb)[None, None] / hparams.cl_temperature

        for other_batch_idx in range(batch_size):
            if other_batch_idx == batch_idx:
                continue
            other_query_emb = embeds["query"][other_batch_idx]
            one_neg_score = (
                similarity_fn(pos_emb, other_query_emb)[None, None] / hparams.cl_temperature
            )
            cur_score = torch.cat([cur_score, one_neg_score], dim=-1)
        if all_another_scores is None:
            all_another_scores = cur_score
        else:
            all_another_scores = torch.cat([all_another_scores, cur_score], dim=0)
    labels_another = torch.zeros(all_another_scores.size(0)).long().to(device)
    loss += F.cross_entropy(all_another_scores, labels_another)
    if hparams.similarity_fn == "cos":
        with torch.no_grad():
            # In the cosine strategy, the minimum possible loss is not zero, when defined as above.
            # For convenience subtract a (batch-size-dependent) constant to make the minimum zero.
            num_cat_all_scores = all_another_scores.shape[-1]
            elems = torch.arange(num_cat_all_scores).to(device)
            perfect_pred = torch.where(
                elems > 0, -1.0 / hparams.cl_temperature, 1.0 / hparams.cl_temperature
            )
            perfect_loss = F.cross_entropy(perfect_pred[None], torch.zeros(1, device=device).long())
        loss -= perfect_loss.item()

    metrics = {"loss": loss}
    return metrics


def train_one_step():
    pass


def validate():
    pass


def get_next_train_inputs(train_dataloader_dict, hparams):
    """Randomly selects a category and returns the next item from the relevant dataloader."""
    num_samples = sum(len(v) for v in train_dataloader_dict.values())
    train_probs = {k: len(v) / num_samples for k, v in train_dataloader_dict.items()}
    task_list, prob_list = [], []
    for k, p in train_probs.items():
        task_list.append(k)
        prob_list.append(p)
    rng = np.random.default_rng(hparams.seed)
    iter_dict = {k: iter(v) for k, v in train_dataloader_dict.items()}

    while True:
        next_cat = rng.choice(task_list, p=prob_list)
        data_iter = iter_dict[next_cat]
        try:
            yield next(data_iter)
        except StopIteration:
            iter_dict[next_cat] = iter(train_dataloader_dict[next_cat])
            data_iter = iter_dict[next_cat]
            yield next(data_iter)


def main(hparams: omegaconf.OmegaConf, core_context: det.core.Context):
    transformers.set_seed(hparams.seed)
    train_dataset_dict, val_dataset_dict = build_train_val_dataset_dicts(hparams)
    train_dataloader_dict, val_dataloader_dict = build_train_val_dataloaders_dict(
        train_dataset_dict, val_dataset_dict, hparams
    )

    small_loader = train_dataloader_dict["zeroshot"]

    train_losses = []
    step = 0

    model, optimizer = build_model_and_optimizer(hparams)
    model.to("cuda")
    for _ in range(hparams.steps):
        model.train()
        for inputs in get_next_train_inputs(train_dataloader_dict, hparams):
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            metrics = process_batch(model, inputs, hparams)
            loss = metrics["loss"]
            loss.backward()
            train_losses.append(loss.item())
            optimizer.step()
            step += 1
            optimizer.zero_grad(set_to_none=True)
            if step % hparams.report_rate == 0:
                mean_train_loss = np.array(train_losses).mean()
                if RANK == 0:
                    if USE_WANDB:
                        wandb.log({"step": step, "train_loss": mean_train_loss})
                    core_context.train.report_training_metrics(
                        steps_completed=step, metrics={"train_loss": mean_train_loss}
                    )
                train_losses = []

        model.eval()
        with torch.no_grad():
            val_losses = []
            for inputs in val_dataloader_dict["zeroshot"]:
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
                metrics = process_batch(model, inputs, hparams)
                loss = metrics["loss"]
                val_losses.append(loss.item())
            mean_val_loss = np.array(val_losses).mean()
            if RANK == 0:
                if USE_WANDB:
                    wandb.log({"step": step, "val_loss": mean_val_loss})
                core_context.train.report_validation_metrics(
                    steps_completed=step, metrics={"val_loss": mean_val_loss}
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
    with det.core.init(distributed=distributed) as core_context:
        if USE_WANDB and core_context.distributed.rank == 0:
            logging.info("Reporting results to wandb ... ")
            assert isinstance(hparams.model_name, str)
            name = (
                hparams.model_name + f"_bsz_{hparams.batch_size}"
                f"_lr_{hparams.lr}"
                + f"_strat_{hparams.embed_strat}"
                + f"_temp_{hparams.cl_temperature}"
                + f"_{hparams.similarity_fn}"
            )
            run = wandb.init(
                project="Embeddings",
                config=dict(hparams),
                name=name,
                job_type="medi_train",
                tags=["train"],
            )

        main(hparams, core_context)
