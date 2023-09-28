import logging
from typing import Dict, List

import determined as det
import omegaconf
import torch
import torch.nn as nn
import transformers


class DetInstructor(nn.Module):
    def __init__(self, hparams: omegaconf.OmegaConf, core_context: det.core.Context) -> None:
        super().__init__()
        self.hf_model = self._build_hf_model(hparams, core_context)
        self.hf_tokenizer = self._build_tokenizer(hparams, core_context)

        self.keys: List[str] = ["query", "pos", "neg"]
        self.embed_strat = hparams.embed_strat

        # GG_TODO: Cover case where loading from checkpoint and hparams.model_name doesn't match.
        if hparams.model_name in ("mosaicml/mpt-7b", "tiiuae/falcon-7b"):
            logging.info("Freezing mpt")
            for n, p in self.hf_model.named_parameters():
                if n.split(".")[1] != "31":
                    logging.info(f"Freezing {n}")
                    p.requires_grad = False

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Takes an input dict with ["pos", "neg", "query"] keys, with dictionary values, each of
        which has `attention_mask` and `input_ids` keys, with tensors as values, and returns a dict
        which embeds each of the ["pos", "neg", "query"] inputs as a fixed-dimensional vector.
        """
        embeds = {}
        for k in self.keys:
            raw_model_outputs = self.hf_model(**inputs[k])
            if self.embed_strat == "last":
                # Need to take the final state according to the attention masks
                attn_mask_sum = inputs[k]["attention_mask"].sum(dim=-1) - 1
                last_hidden_state = raw_model_outputs.last_hidden_state
                embeds[k] = last_hidden_state[range(last_hidden_state.shape[0]), attn_mask_sum]
            else:
                raise ValueError(f"Received unexpected strat {self.embed_strat}")
        return embeds

    def save_pretrained(self, *args, **kwargs):
        self.hf_model.save_pretrained(*args, **kwargs)
        self.hf_tokenizer.save_pretrained(*args, **kwargs)

    def _build_hf_model(
        self, hparams: omegaconf.OmegaConf, core_context
    ) -> transformers.PreTrainedModel:
        if not hparams.get("load_from_uuid"):
            logging.info("Loading model from the HF Hub")
            hf_model = transformers.AutoModel.from_pretrained(
                hparams.model_name, trust_remote_code=hparams.model_name == "tiiuae/falcon-7b"
            )
        else:
            with core_context.checkpoint.restore_path(hparams.load_from_uuid) as path:
                logging.info(f"Loading model from the checkpoint {hparams.load_from_uuid}")
                hf_model = transformers.AutoModel.from_pretrained(
                    path, trust_remote_code=hparams.model_name == "tiiuae/falcon-7b"
                )

        return hf_model

    def _build_tokenizer(self, hparams: omegaconf.OmegaConf, core_context: det.core.Context):
        if not hparams.get("load_from_uuid"):
            logging.info("Loading tokenizer from the HF Hub")
            tokenizer_name = hparams.get("tokenizer_name") or hparams.model_name
            tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            with core_context.checkpoint.restore_path(hparams.load_from_uuid) as path:
                logging.info(f"Loading tokenizer from the checkpoint {hparams.load_from_uuid}")
                tokenizer = transformers.AutoTokenizer.from_pretrained(path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.paddding_side = "right"
        return tokenizer
