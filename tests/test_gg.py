from sentence_transformers import SentenceTransformer

from mteb import MTEB


def test_basic_results():
    # Define the sentence-transformers model name
    model_name = "average_word_embeddings_komninos"

    model = SentenceTransformer(model_name)
    evaluation = MTEB(tasks=["Banking77Classification"])
    results = evaluation.run(model, output_folder=f"test_results/{model_name}")
    assert results
