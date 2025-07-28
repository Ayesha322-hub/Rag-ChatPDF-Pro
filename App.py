import gradio as gr
import os
import requests
import fitz  # PyMuPDF
import tempfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# GROQ API setup
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")  # Or manually set it like: "your-key-here"
MODEL_NAME = "llama3-8b-8192"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Extract text and chunks from PDFs
def extract_chunks_from_pdfs(file_objs, chunk_size=1000):
    full_text = ""
    for file_obj in file_objs:
        with open(file_obj.name, "rb") as f:
            doc = fitz.open(stream=f.read(), filetype="pdf")
            for page in doc:
                full_text += page.get_text()
    chunks = [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size)]
    return chunks, full_text

# TF-IDF similarity-based retrieval
def get_top_chunks(chunks, question, top_n=3):
    vectorizer = TfidfVectorizer().fit(chunks + [question])
    vectors = vectorizer.transform(chunks + [question])
    similarities = cosine_similarity(vectors[-1], vectors[:-1]).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    return [chunks[i] for i in top_indices]

# Generate summary using LLM
def generate_summary_text(files):
    if not files:
        return "Please upload PDFs first."
    
    _, full_text = extract_chunks_from_pdfs(files)
    short_text = full_text[:2000]  # To stay within prompt length

    prompt = f"Summarize the following content:\n\n{short_text}"
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful summarizer."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }

    response = requests.post(GROQ_API_URL, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error generating summary: {response.status_code} - {response.text}"

# Ask question and keep chat history
def ask_question_with_history(files, question, history):
    if not files or not question:
        return "Please upload PDFs and enter a question.", history, ""

    try:
        chunks, _ = extract_chunks_from_pdfs(files)
        top_chunks = get_top_chunks(chunks, question)
        context = "\n\n".join(top_chunks)

        prompt = f"Use the following context to answer the question:\n\n{context}\n\nQuestion: {question}"

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        data = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant who only answers using the provided context."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3
        }

        response = requests.post(GROQ_API_URL, headers=headers, json=data)

        if response.status_code == 200:
            answer = response.json()["choices"][0]["message"]["content"]
            history += f"\n\nUser: {question}\nAssistant: {answer}"
            return answer, history, ""
        else:
            error = f"Error {response.status_code}: {response.text}"
            return error, history, ""
    
    except Exception as e:
        return f"Exception occurred: {str(e)}", history, ""

# Save chat history
def download_chat_history(history):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as f:
        f.write(history)
        return f.name

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 📄 RAG Chatbot with PDF Preview & Chat History Export")

    history_state = gr.State("")

    with gr.Row():
        files = gr.File(label="Upload PDF files", file_types=[".pdf"], file_count="multiple")
        summary_btn = gr.Button("📘 Preview PDF Summary")

    summary_output = gr.Textbox(label="PDF Content Preview", lines=10)
    summary_btn.click(fn=generate_summary_text, inputs=files, outputs=summary_output)

    question = gr.Textbox(label="Enter your question")
    answer_output = gr.Textbox(label="Answer", lines=5)

    ask_btn = gr.Button("❓ Ask")
    ask_btn.click(fn=ask_question_with_history, inputs=[files, question, history_state], outputs=[answer_output, history_state, gr.Textbox(visible=False)])

    with gr.Row():
        download_btn = gr.Button("⬇️ Download Chat History")
        download_file = gr.File(label="Download", interactive=False)

    download_btn.click(fn=download_chat_history, inputs=history_state, outputs=download_file)

demo.launch()
