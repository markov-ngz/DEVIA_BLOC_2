import pytest
from evaluate_model import compute_bleu, get_bleu_score


def test_compute_bleu(test_data,model_and_tokenizer):

    tokenizer, model = model_and_tokenizer

    all_label, predictions = compute_bleu(test_data, model, tokenizer)

    # Assertions based on expected output format and data
    assert len(all_label) == len(predictions)  # Check equal lengths
    for label, prediction in zip(all_label, predictions):
        assert isinstance(label, str)  # Check label type (string)
        assert isinstance(prediction, str)  # Check prediction type (string)


def test_get_bleu_score(test_data,model_and_tokenizer):

    tokenizer, model = model_and_tokenizer
    
    bleu_score = get_bleu_score(test_data, model, tokenizer)

    # Assertions based on BLEU score structure (dictionary)
    assert isinstance(bleu_score, dict)
    assert "score" in bleu_score.keys()  # Check presence of 'score' key
    assert bleu_score["score"] > 0  # Check positive score
