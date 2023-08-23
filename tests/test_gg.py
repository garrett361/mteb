import time

from sentence_transformers import SentenceTransformer

from mteb import MTEB


def test_basic_results():
    # Define the sentence-transformers model name
    model_name = "average_word_embeddings_komninos"

    model = SentenceTransformer(model_name)
    evaluation = MTEB(tasks=["Banking77Classification"])
    # Create a unique output folder; the evaluation exits early if the output dir already exists
    output_folder = f"tests/test_results/{model_name}_" + f"{str(int(time.time()))}"
    results = evaluation.run(model, output_folder=output_folder)
    assert results
