from transformers import pipeline

def classify_text(text: str):
    # Load a pre-trained sentiment analysis pipeline
    classifier = pipeline("sentiment-analysis")
    # Perform classification
    result = classifier(text)
    return result

if __name__ == "__main__":
    input_text = "Hugging Face is amazing!"
    sentiment = classify_text(input_text)
    print(f"Input: {input_text}")
    print(f"Sentiment: {sentiment}")
