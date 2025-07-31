from transformers import pipeline

# Existing QA pipeline
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

# ✅ New summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def generate_answer(query, contexts):
    context_text = "\n".join(contexts)
    prompt = f"Context: {context_text}\nQuestion: {query}"
    response = qa_pipeline(prompt, max_new_tokens=256)[0]["generated_text"]
    return response

# ✅ Add this new function
def summarize_text(text):
    summary = summarizer(text, max_length=512, min_length=60, do_sample=False)
    return summary[0]["summary_text"]
