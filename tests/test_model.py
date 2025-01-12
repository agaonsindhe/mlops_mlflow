from src.model import classify_text

def test_classify_text():
    text = "This is great!"
    result = classify_text(text)
    assert len(result) > 0, "No classification result returned!"
    assert "label" in result[0], "Missing 'label' in result!"
    assert "score" in result[0], "Missing 'score' in result!"
