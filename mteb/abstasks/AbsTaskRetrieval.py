import logging
import os
from time import time
from typing import Dict, List

from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, WordEmbeddings

from .AbsTask import AbsTask

logger = logging.getLogger(__name__)

DRES_METHODS = ["encode_queries", "encode_corpus"]


class AbsTaskRetrieval(AbsTask):
    """
    Abstract class for re-ranking experiments.
    Child-classes must implement the following properties:
    self.corpus = Dict[id, Dict[str, str]] #id => dict with document datas like title and text
    self.queries = Dict[id, str] #id => query
    self.relevant_docs = List[id, id, score]
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def is_dres_compatible(model):
        for method in DRES_METHODS:
            op = getattr(model, method, None)
            if not (callable(op)):
                return False
        return True

    def evaluate(
        self, model, split="test", batch_size=128, corpus_chunk_size=None, score_function="cos_sim", **kwargs
    ):
        logging.info(80 * "*")
        logging.info(f"In evaluate, using batch_size {batch_size}")
        logging.info(80 * "*")
        try:
            from beir.retrieval.evaluation import EvaluateRetrieval
        except ImportError:
            raise Exception(
                "Retrieval tasks require beir package. Please install it with `pip install mteb[beir]`"
            )

        if not self.data_loaded:
            self.load_data()

        corpus, queries, relevant_docs = self.corpus[split], self.queries[split], self.relevant_docs[split]
        if not self.is_dres_compatible(model):
            logging.info("Model not DRES compatible; initializing DRESModel")
            model = DRESModel(model, task_name=self.description["name"], **kwargs)
        else:
            logging.info("Model DRES compatible")

        # assert hasattr(model, "start_multi_process_pool"), "No start_multi_process_pool attr after first DRES"
        if os.getenv("RANK", None) is None:
            # Non-distributed
            from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

            # assert hasattr(model, "start_multi_process_pool"), "No start_multi_process_pool attr before DRES"
            model = DRES(
                model,
                batch_size=batch_size,
                corpus_chunk_size=corpus_chunk_size if corpus_chunk_size is not None else 50000,
                **kwargs,
            )
            # assert hasattr(model, "start_multi_process_pool"), "No start_multi_process_pool attr after DRES"

        else:
            logging.info(80 * "*")
            logging.info("In DIST block")
            logging.info(80 * "*")
            # Distributed (multi-GPU)
            from beir.retrieval.search.dense import (
                DenseRetrievalParallelExactSearch as DRPES,
            )

            # assert hasattr(model, "start_multi_process_pool"), "No start_multi_process_pool attr before DRPES"
            model = DRPES(
                model,
                batch_size=batch_size,
                corpus_chunk_size=corpus_chunk_size,
                **kwargs,
            )
            # assert hasattr(model, "start_multi_process_pool"), "No start_multi_process_pool attr after DRPES"

        retriever = EvaluateRetrieval(model, score_function=score_function)  # or "cos_sim" or "dot"
        start_time = time()
        results = retriever.retrieve(corpus, queries)
        end_time = time()
        logger.info("Time taken to retrieve: {:.2f} seconds".format(end_time - start_time))

        ndcg, _map, recall, precision = retriever.evaluate(
            relevant_docs,
            results,
            retriever.k_values,
            ignore_identical_ids=kwargs.get("ignore_identical_ids", True),
        )
        mrr = retriever.evaluate_custom(relevant_docs, results, retriever.k_values, "mrr")

        scores = {
            **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
            **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
            **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
            **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
            **{f"mrr_at_{k.split('@')[1]}": v for (k, v) in mrr.items()},
        }

        return scores


class DRESModel:
    """
    Dense Retrieval Exact Search (DRES) in BeIR requires an encode_queries & encode_corpus method.
    This class converts a MTEB model (with just an .encode method) into BeIR DRES format.
    """

    def __init__(self, model, sep=" ", **kwargs):
        self.model = model
        self.sep = sep
        self.use_sbert_model = isinstance(model, SentenceTransformer)
        self.model_type = kwargs["hparams"]["model_type"]
        self.model_name = kwargs["hparams"]["model_name"]
        self.task_name = kwargs["task_name"]
        self.model_key = (
            self.model_name
            if self.model_name in ["hkunlp/instructor-xl", "hkunlp/instructor-base"]
            else "hkunlp/instructor-large"
        )

    def encode_queries(self, queries: List[str], batch_size: int, **kwargs):
        if self.use_sbert_model:
            if isinstance(self.model._first_module(), Transformer):
                logger.info(f"Queries will be truncated to {self.model.get_max_seq_length()} tokens.")
            elif isinstance(self.model._first_module(), WordEmbeddings):
                logger.warning(
                    "Queries will not be truncated. This could lead to memory issues. In that case please lower the batch_size."
                )
        assert self.model_type in ("INSTRUCTOR", "SentenceTransformer")
        if self.model_type == "SentenceTransformer":
            return self.model.encode(queries, batch_size=batch_size, **kwargs)
        elif self.model_type == "INSTRUCTOR":
            new_sentences = []

            instruction = DEFINITIONS[self.model_key][self.task_name]["query"]
            logging.info(f"Using instruction {instruction} for INSTRUCTOR query encoding")
            for s in queries:
                # GG_NOTE: Why the zero at the end? Removing it does not seem to change anything.
                new_sentences.append([instruction, s, 0])
            return self.model.encode(new_sentences, batch_size=batch_size, **kwargs)
        else:
            raise ValueError(f"Unexpected model type: {self.model_type}")

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs):
        if self.model_type == "SentenceTransformer":
            if type(corpus) is dict:
                sentences = [
                    (corpus["title"][i] + self.sep + corpus["text"][i]).strip()
                    if "title" in corpus
                    else corpus["text"][i].strip()
                    for i in range(len(corpus["text"]))
                ]
            else:
                sentences = [
                    (doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip()
                    for doc in corpus
                ]
            return self.model.encode(sentences, batch_size=batch_size, **kwargs)
        elif self.model_type == "INSTRUCTOR":
            if type(corpus) is dict:
                sentences = [
                    (corpus["title"][i] + " " + corpus["text"][i]).strip()
                    if "title" in corpus
                    else corpus["text"][i].strip()
                    for i in range(len(corpus["text"]))
                ]
            else:
                sentences = [
                    (doc["title"] + " " + doc["text"]).strip() if "title" in doc else doc["text"].strip()
                    for doc in corpus
                ]
            new_sentences = []
            instruction = DEFINITIONS[self.model_key][self.task_name]["corpus"]
            logging.info(f"Using instruction {instruction} for INSTRUCTOR corpus encoding")
            for s in sentences:
                new_sentences.append([instruction, s, 0])
            return self.model.encode(sentences, batch_size=128, **kwargs)
        else:
            raise ValueError(f"Unexpected model type: {self.model_type}")


DEFINITIONS = {
    "hkunlp/instructor-xl": {
        "ClimateFEVER": {
            "query": "Represent the Climate question for retrieving supporting documents: ",
            "corpus": "Represent the document for retrieval: ",
        },
        "HotpotQA": {
            "query": "Represent the Wikipedia question for retrieving supporting documents: ",
            "corpus": "Represent the document for retrieval: ",
        },
        "FEVER": {
            "query": "Represent the fact for retrieving supporting evidence: ",
            "corpus": "Represent the evidence for retrieval: ",
        },
        "MSMARCO": {
            "query": "Represent the question for retrieving supporting documents: ",
            "corpus": "Represent the document for retrieval: ",
        },
        "DBPedia": {
            "query": "Represent the Wikipedia questions to retrieve a supporting document: ",
            "corpus": "Represent the Wikipedia documents for retrieval: ",
        },
        "NQ": {
            "query": "Represent the Wikipedia question for retrieving supporting documents: ",
            "corpus": "Represent the document for retrieval: ",
        },
        "QuoraRetrieval": {
            "query": "Represent the Quora question to retrieve question: ",
            "corpus": "Represent the Quora question to retrieve question: ",
        },
        "SCIDOCS": {
            "query": "Represent a Science question for retrieving supporting papers: ",
            "corpus": "Represent the Science paper: ",
        },
        "TRECCOVID": {
            "query": "Represent the Coronavirus questions to retrieve a supporting document: ",
            "corpus": "Represent the Coronavirus documents for retrieval: ",
        },
        "Touche2020": {
            "query": "Represent questions: ",
            "corpus": "Represent arguments: ",
        },
        "SciFact": {
            "query": "Represent the Scientific queries for retrieving a supporting passage: ",
            "corpus": "represent the scientific paragraph for retrieval: ",
        },
        "NFCorpus": {
            "query": "Represent the nutrition facts to retrieve Public medical articles: ",
            "corpus": "Represent the Public medical articles for retrieval: ",
        },
        "ArguAna": {
            "query": "Represent Debating conversations to retrieve a counter-argument: ",
            "corpus": "Represent counter-arguments: ",
        },
        "CQADupstackTexRetrieval": {
            "query": "Represent the question for retrieving answers: ",
            "corpus": "Represent the answer for retrieval: ",
        },
        "CQADupstackWebmastersRetrieval": {
            "query": "Represent the Webmaster question for retrieving answers: ",
            "corpus": "Represent the Webmaster answer: ",
        },
        "CQADupstackEnglishRetrieval": {
            "query": "Represent the English question for retrieving documents: ",
            "corpus": "Represent the English answer for retrieval: ",
        },
        "CQADupstackGamingRetrieval": {
            "query": "Represent the Gaming question for retrieving answers: ",
            "corpus": "Represent the Gaming answer for retrieval: ",
        },
        "CQADupstackGisRetrieval": {
            "query": "Represent the Gis question for retrieving answers: ",
            "corpus": "Represent the Gis answer for retrieval: ",
        },
        "CQADupstackUnixRetrieval": {
            "query": "Represent the Unix questions to retrieve a supporting answer: ",
            "corpus": "Represent the Unix answers for retrieval: ",
        },
        "CQADupstackMathematicaRetrieval": {
            "query": "Represent the Mathematical question for retrieving answers: ",
            "corpus": "Represent the Mathematical answer for retrieval: ",
        },
        "CQADupstackStatsRetrieval": {
            "query": "Represent the Statistical question for retrieving answers: ",
            "corpus": "Represent the Statistical answer for retrieval: ",
        },
        "CQADupstackPhysicsRetrieval": {
            "query": "Represent the Physics question for retrieving answers: ",
            "corpus": "Represent the Physics answer for retrieval: ",
        },
        "CQADupstackProgrammersRetrieval": {
            "query": "Represent the Programming question for retrieving answers: ",
            "corpus": "Represent the Programming answer for retrieval: ",
        },
        "CQADupstackAndroidRetrieval": {
            "query": "Represent the Android question for retrieving answers: ",
            "corpus": "Represent the Android answer for retrieval: ",
        },
        "CQADupstackWordpressRetrieval": {
            "query": "Represent the Wordpress question for retrieving answers: ",
            "corpus": "Represent the Wordpress answer for retrieval: ",
        },
        "FiQA2018": {
            "query": "Represent the finance questions to retrieve a supporting answer: ",
            "corpus": "Represent the finance answers for retrieval: ",
        },
    },
    "hkunlp/instructor-large": {
        "ClimateFEVER": {
            "query": "Represent the Climate question for retrieving supporting documents: ",
            "corpus": "Represent the document for retrieval: ",
        },
        "HotpotQA": {
            "query": "Represent the Wikipedia question for retrieving supporting documents: ",
            "corpus": "Represent the document for retrieval: ",
        },
        "FEVER": {
            "query": "Represent the fact for retrieving supporting evidence: ",
            "corpus": "Represent the evidence for retrieval: ",
        },
        "MSMARCO": {
            "query": "Represent the question for retrieving supporting documents: ",
            "corpus": "Represent the document for retrieval: ",
        },
        "DBPedia": {
            "query": "Represent the Wikipedia sentence for retrieving supporting documents: ",
            "corpus": "Represent the document for retrieval: ",
        },
        "NQ": {
            "query": "Represent the Wikipedia question for retrieving supporting documents: ",
            "corpus": "Represent the document for retrieval: ",
        },
        "QuoraRetrieval": {
            "query": "Represent the Quora question for retrieving duplicate questions: ",
            "corpus": "Represent the Quora question for retrieving duplicate questions: ",
        },
        "SCIDOCS": {
            "query": "Represent a Science question for retrieving supporting papers: ",
            "corpus": "Represent the Science paper: ",
        },
        "TRECCOVID": {
            "query": "Represent the Coronavirus question for retrieving supporting documents: ",
            "corpus": "Represent the Coronavirus document for retrieval: ",
        },
        "Touche2020": {
            "query": "Represent a question: ",
            "corpus": "Represent an argument: ",
        },
        "SciFact": {
            "query": "Represent a Scientific query for retrieving a supporting passage; ",
            "corpus": "represent the Scientific passage for retrieval; ",
        },
        "NFCorpus": {
            "query": "Represent the Medicine question for retrieving a relevant document: ",
            "corpus": "Represent the medical document for retrieval: ",
        },
        "ArguAna": {
            "query": "Represent a Debate argument for retrieving a counter-argument: ",
            "corpus": "Represent a Counter-argument: ",
        },
        "CQADupstackTexRetrieval": {
            "query": "Represent the question for retrieving answers: ",
            "corpus": "Represent the answer for retrieval: ",
        },
        "CQADupstackWebmastersRetrieval": {
            "query": "Represent the Webmaster question for retrieving answers: ",
            "corpus": "Represent the Webmaster answer: ",
        },
        "CQADupstackEnglishRetrieval": {
            "query": "Represent the English question for retrieving documents: ",
            "corpus": "Represent the English answer for retrieval: ",
        },
        "CQADupstackGamingRetrieval": {
            "query": "Represent the Gaming question for retrieving answers: ",
            "corpus": "Represent the Gaming answer for retrieval: ",
        },
        "CQADupstackGisRetrieval": {
            "query": "Represent the Gis question for retrieving answers: ",
            "corpus": "Represent the Gis answer for retrieval: ",
        },
        "CQADupstackUnixRetrieval": {
            "query": "Represent the Unix question for retrieving answers: ",
            "corpus": "Represent the Unix answer for retrieval: ",
        },
        "CQADupstackMathematicaRetrieval": {
            "query": "Represent the Mathematical question for retrieving answers: ",
            "corpus": "Represent the Mathematical answer for retrieval: ",
        },
        "CQADupstackStatsRetrieval": {
            "query": "Represent the Statistical question for retrieving answers: ",
            "corpus": "Represent the Statistical answer for retrieval: ",
        },
        "CQADupstackPhysicsRetrieval": {
            "query": "Represent the Physics question for retrieving answers: ",
            "corpus": "Represent the Physics answer for retrieval: ",
        },
        "CQADupstackProgrammersRetrieval": {
            "query": "Represent the Programming question for retrieving answers: ",
            "corpus": "Represent the Programming answer for retrieval: ",
        },
        "CQADupstackAndroidRetrieval": {
            "query": "Represent the Android question for retrieving answers: ",
            "corpus": "Represent the Android answer for retrieval: ",
        },
        "CQADupstackWordpressRetrieval": {
            "query": "Represent the Wordpress question for retrieving answers: ",
            "corpus": "Represent the Wordpress answer for retrieval: ",
        },
        "FiQA2018": {
            "query": "Represent the finance question for retrieving the supporting answers: ",
            "corpus": "Represent the finance answer for retrieval: ",
        },
    },
    "hkunlp/instructor-base": {
        "ClimateFEVER": {
            "query": "Represent the Climate question for retrieving supporting documents: ",
            "corpus": "Represent the document for retrieval: ",
        },
        "HotpotQA": {
            "query": "Represent the Wikipedia question for retrieving supporting documents: ",
            "corpus": "Represent the document for retrieval: ",
        },
        "FEVER": {
            "query": "Represent the fact for retrieving supporting evidence: ",
            "corpus": "Represent the evidence for retrieval: ",
        },
        "MSMARCO": {
            "query": "Represent the question for retrieving supporting documents: ",
            "corpus": "Represent the document for retrieval: ",
        },
        "DBPedia": {
            "query": "Represent the Wikipedia sentence for retrieving supporting documents: ",
            "corpus": "Represent the document for retrieval: ",
        },
        "NQ": {
            "query": "Represent the Wikipedia question for retrieving supporting documents: ",
            "corpus": "Represent the document for retrieval: ",
        },
        "QuoraRetrieval": {
            "query": "Represent the Quora question for retrieving duplicate questions: ",
            "corpus": "Represent the Quora question for retrieving duplicate questions: ",
        },
        "SCIDOCS": {
            "query": "Represent a Science question for retrieving supporting papers: ",
            "corpus": "Represent the Science paper: ",
        },
        "TRECCOVID": {
            "query": "Represent the Coronavirus question for retrieving supporting documents: ",
            "corpus": "Represent the Coronavirus document for retrieval: ",
        },
        "Touche2020": {
            "query": "Represent a question: ",
            "corpus": "Represent an argument: ",
        },
        "SciFact": {
            "query": "Represent a Scientific query for retrieving a supporting passage; ",
            "corpus": "represent the Scientific passage for retrieval; ",
        },
        "NFCorpus": {
            "query": "Represent the Medicine question for retrieving a relevant document: ",
            "corpus": "Represent the medical document for retrieval: ",
        },
        "ArguAna": {
            "query": "Represent the Debate argument for retrieving a counter-argument: ",
            "corpus": "Represent the Counter debate argument: ",
        },
        "CQADupstackTexRetrieval": {
            "query": "Represent the question for retrieving answers: ",
            "corpus": "Represent the answer for retrieval: ",
        },
        "CQADupstackWebmastersRetrieval": {
            "query": "Represent the Webmaster question for retrieving answers: ",
            "corpus": "Represent the Webmaster answer: ",
        },
        "CQADupstackEnglishRetrieval": {
            "query": "Represent the English question for retrieving documents: ",
            "corpus": "Represent the English answer for retrieval: ",
        },
        "CQADupstackGamingRetrieval": {
            "query": "Represent the Gaming question for retrieving answers: ",
            "corpus": "Represent the Gaming answer for retrieval: ",
        },
        "CQADupstackGisRetrieval": {
            "query": "Represent the Gis question for retrieving answers: ",
            "corpus": "Represent the Gis answer for retrieval: ",
        },
        "CQADupstackUnixRetrieval": {
            "query": "Represent the Unix question for retrieving answers: ",
            "corpus": "Represent the Unix answer for retrieval: ",
        },
        "CQADupstackMathematicaRetrieval": {
            "query": "Represent the Mathematical question for retrieving answers: ",
            "corpus": "Represent the Mathematical answer for retrieval: ",
        },
        "CQADupstackStatsRetrieval": {
            "query": "Represent the Statistical question for retrieving answers: ",
            "corpus": "Represent the Statistical answer for retrieval: ",
        },
        "CQADupstackPhysicsRetrieval": {
            "query": "Represent the Physics question for retrieving answers: ",
            "corpus": "Represent the Physics answer for retrieval: ",
        },
        "CQADupstackProgrammersRetrieval": {
            "query": "Represent the Programming question for retrieving answers: ",
            "corpus": "Represent the Programming answer for retrieval: ",
        },
        "CQADupstackAndroidRetrieval": {
            "query": "Represent the Android question for retrieving answers: ",
            "corpus": "Represent the Android answer for retrieval: ",
        },
        "CQADupstackWordpressRetrieval": {
            "query": "Represent the Wordpress question for retrieving answers: ",
            "corpus": "Represent the Wordpress answer for retrieval: ",
        },
        "FiQA2018": {
            "query": "Represent the finance question for retrieving the supporting answers: ",
            "corpus": "Represent the finance answer for retrieval: ",
        },
    },
}
