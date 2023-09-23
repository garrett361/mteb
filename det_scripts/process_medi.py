import contextlib
import json
import logging
import os

import datasets
import determined as det
import omegaconf
import torch
import transformers


@contextlib.contextmanager
def local_rank_zero_first():
    world_size = int(os.getenv("WORLD_SIZE", 1))
    if world_size > 1:
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        is_local_rank_zero = local_rank == 0
        try:
            if not is_local_rank_zero:
                torch.distributed.barrier()
            yield
        finally:
            if is_local_rank_zero:
                torch.distributed.barrier()
    else:
        yield


def main(hparams: omegaconf.OmegaConf, core_context: det.core.Context):

    with open(hparams.path_to_medi_json, "r") as f:
        train_examples_raw = json.load(f)

    total_train_n = len(train_examples_raw)
    world_size = int(os.getenv("WORLD_SIZE", 1))
    global_batch_size = hparams.per_device_train_batch_size * world_size

    def batch_raw_examples_by_task(old_examples_raw, total_n, global_batch_size):
        """Create `global_batch_size` batches of examples from the same task."""
        examples_raw = []
        for idx in range(0, total_n, global_batch_size):
            local_task_name = old_examples_raw[idx]["task_name"]
            cur_batch = []
            include_batch = True
            for idx1 in range(idx, min(idx + global_batch_size, total_n)):
                if not old_examples_raw[idx1]["task_name"] == local_task_name:
                    print(f'one batch in task {old_examples_raw[idx1]["task_name"]} is skipped')
                    include_batch = False
                    break
                else:
                    cur_batch.append(old_examples_raw[idx1])
            if include_batch and len(cur_batch) == global_batch_size:
                examples_raw.append(cur_batch)
        return examples_raw

    train_examples_raw_batched = batch_raw_examples_by_task(
        train_examples_raw, total_train_n, global_batch_size
    )

    def get_dataset(examples_raw_batched):
        examples = {"query": [], "pos": [], "neg": [], "task_name": []}
        task_name_map = {}
        total_num = len(examples_raw_batched)
        task_count = 0
        for i in range(total_num):
            logging.info(f"examples_raw_batched: {examples_raw_batched}")
            cur_e = examples_raw_batched[i]
            logging.info(f"cur_e: {cur_e}")
            for k in ["query", "pos", "neg"]:
                for s in cur_e[k][:-1]:
                    assert "!@#$%^&**!@#$%^&**" not in s
                cur_e[k][-1] = str(cur_e[k][-1])
                assert cur_e[k][0].startswith("Represent ") or cur_e[k][0] == ""
                examples[k].append("!@#$%^&**!@#$%^&**".join(cur_e[k]))
            if cur_e["task_name"] not in task_name_map:
                task_name_map[cur_e["task_name"]] = task_count
                task_count += 1
            examples["task_name"].append(task_name_map[cur_e["task_name"]])
        return examples

    raw_batched_medi_dataset = datasets.Dataset.from_dict(get_dataset(train_examples_raw_batched))
    print(raw_batched_medi_dataset[0])
    if hparams.seed:
        raw_batched_medi_dataset = raw_batched_medi_dataset.shuffle(seed=hparams.seed)

    tokenizer = transformers.AutoTokenizer.from_pretrained(hparams.model_name)

    def preprocess_function(examples):
        all_tokenized = None
        for key in ["query", "pos", "neg"]:
            num = len(examples[key])
            contexts = []
            concatenated_input_texts = []
            for local_idx in range(num):
                splits = examples[key][local_idx].split("!@#$%^&**!@#$%^&**")
                assert len(splits) == 2
                contexts.append(splits[0])
                concatenated_input_texts.append("".join(splits))
                assert isinstance(contexts[-1], str)
                assert isinstance(concatenated_input_texts[-1], str)
            tokenized = tokenizer(
                concatenated_input_texts,
                padding="max_length",
                truncation="longest_first",
                return_tensors="pt",
                max_length=hparams.max_source_length,
            )
            context_tok = tokenizer(
                contexts,
                padding="max_length",
                truncation="longest_first",
                return_tensors="pt",
                max_length=hparams.max_source_length,
            )
            tokenized["context_masks"] = torch.sum(context_tok["attention_mask"], dim=1)
            tokenized["context_masks"] = tokenized["context_masks"] - 1
            for my_idx in range(len(tokenized["context_masks"])):
                if tokenized["context_masks"][my_idx] <= 1:
                    tokenized["context_masks"][my_idx] = 0
            keys = tokenized.keys()
            if all_tokenized is None:
                all_tokenized = tokenized.copy()
                for k in keys:
                    all_tokenized[k] = all_tokenized[k].tolist()
            for k in keys:
                all_tokenized[f"{key}_{k}"] = tokenized[k].tolist()
        all_tokenized["task_name"] = examples["task_name"]
        return all_tokenized

    with local_rank_zero_first():
        cpus = os.cpu_count()
        assert isinstance(cpus, int)
        medi_dataset = raw_batched_medi_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=cpus // 2,
            remove_columns=raw_batched_medi_dataset.column_names,
            desc="Running tokenizer on train dataset",
        )
        medi_dataset.save_to_disk(hparams.dataset_save_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=det.LOG_FORMAT)
    info = det.get_cluster_info()
    assert info, "This script must run on a determined cluster."
    hparams = omegaconf.OmegaConf.create(info.trial.hparams)
    try:
        distributed = det.core.DistributedContext.from_torch_distributed()
    except KeyError:
        distributed = None
    with det.core.init(distributed=distributed) as core_context:
        main(hparams, core_context)
