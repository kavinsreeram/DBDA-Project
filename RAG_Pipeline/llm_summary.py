from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

MODEL_NAME = "google/flan-t5-small"
device = torch.device("cpu")

def load_model():
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()


def generate_news_summary(claim, top_articles):

    if not top_articles:
        return "No reliable articles were found to generate a summary."

    combined_context = ""

    for i, article in enumerate(top_articles, 1):
        text_snippet = article["text"][:400]
        combined_context += f"\nArticle {i}:\n{text_snippet}\n"

    prompt = f"""
Summarize the following news articles related to the claim.

Claim:
{claim}

Articles:
{combined_context}

Write one concise and factual paragraph.
Do not speculate.
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=384
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=120,
            min_length=50,
            num_beams=2,
            early_stopping=True
        )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    return summary
