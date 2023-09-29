import logging
from typing import Dict, List

import determined as det
import omegaconf
import sentence_transformers
import torch
import torch.nn as nn
import transformers

TRUST_REMOTE_CODE_MODELS = ("mosaicml/mpt-1b-redpajama-200b", "tiiuae/falcon-7b")
FREEZE_LAYER_MODELS = ("mosaicml/mpt-7b", "tiiuae/falcon-7b")


class DetInstructor(nn.Module):
    def __init__(self, hparams: omegaconf.OmegaConf, core_context: det.core.Context) -> None:
        super().__init__()
        self.hparams = hparams
        self.hf_model = self._build_hf_model(hparams, core_context)
        self.hf_tokenizer = self._build_tokenizer(hparams, core_context)
        self.keys: List[str] = ["query", "pos", "neg"]
        self.pooling_mode = hparams.pooling_mode
        try:
            self.hidden_size = self.hf_model.config.hidden_size
        except AttributeError:
            self.hidden_size = self.hf_model.config.d_model
        except AttributeError:
            raise AttributeError(
                "Model config has neither `hidden_size` nor `d_model`, need to know hidden size."
            )

        self.pool = sentence_transformers.models.Pooling(
            self.hidden_size, pooling_mode=self.pooling_mode
        )
        # GG_TODO: Cover case where loading from checkpoint and hparams.model_name doesn't match.
        if hparams.model_name in FREEZE_LAYER_MODELS:
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
            raw_model_outputs = self.hf_model(**inputs[k], output_hidden_states=True)
            last_hidden_state = raw_model_outputs.hidden_states[-1]
            pool_inputs = {
                "token_embeddings": last_hidden_state,
                "attention_mask": inputs[k]["attention_mask"],
            }
            embeds[k] = self.pool(pool_inputs)["sentence_embedding"]
        return embeds

    # def encode(self, sentences, batch_size: int = 32, **kwargs) -> List[torch.Tensor]:
    #     instructor_sentences = self._create_instructor_sentences(sentences, **kwargs)
    #     tokenized_instructor_sentences = self.hf_tokenizer()
    #     encoded_instructor = self._create_instructor_sentences(sentences, **kwargs)
    #     return torch.randn(len(sentences), 128, device=self.device)

    def save_pretrained(self, *args, **kwargs):
        self.hf_model.save_pretrained(*args, **kwargs)
        self.hf_tokenizer.save_pretrained(*args, **kwargs)

    def _build_hf_model(
        self, hparams: omegaconf.OmegaConf, core_context
    ) -> transformers.PreTrainedModel:
        if not hparams.get("load_from_uuid"):
            logging.info("Loading model from the HF Hub")
            hf_model = transformers.AutoModelForCausalLM.from_pretrained(
                hparams.model_name,
                trust_remote_code=hparams.model_name in TRUST_REMOTE_CODE_MODELS,
            )
        else:
            with core_context.checkpoint.restore_path(hparams.load_from_uuid) as path:
                logging.info(f"Loading model from the checkpoint {hparams.load_from_uuid}")
                hf_model = transformers.AutoModelForCausalLM.from_pretrained(
                    path, trust_remote_code=hparams.model_name in TRUST_REMOTE_CODE_MODELS
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
