from flask import Flask, render_template, request
import os

from pypdf import PdfReader
from openai import OpenAI

phi_client = OpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio" 
)

def extract_pdf_text_fast(pdf_path, max_pages=3):
    """
    Fast text extraction using pypdf.
    Limits pages to keep things fast.
    """
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for i, page in enumerate(reader.pages):
            if i >= max_pages:
                break
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception:
        pass

    return text.strip()


def answer_with_phi3(document_text, question):
    document_text = document_text[:3000]

    messages = [
        {
            "role": "system",
            "content": (
                "You are an internal company assistant. "
                "Answer ONLY using the provided document. "
                "If the answer is not present, reply exactly: "
                "'Not mentioned in the document.'"
            )
        },
        {
            "role": "user",
            "content": f"Document:\n{document_text}\n\nQuestion:\n{question}"
        }
    ]

    response = phi_client.chat.completions.create(
        model="phi-3-mini-4k-instruct",
        messages=messages,
        temperature=0.0 
    )

    return response.choices[0].message.content.strip()


app = Flask(__name__, template_folder="templates")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

cached_document_text = ""


@app.route("/", methods=["GET", "POST"])
def index():
    global cached_document_text
    answer = ""

    if request.method == "POST":
        files = request.files.getlist("documents")
        question = request.form.get("question")

        if files and files[0].filename:
            cached_document_text = ""

            for file in files:
                if file and file.filename.lower().endswith(".pdf"):
                    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
                    file.save(file_path)

                    cached_document_text += extract_pdf_text_fast(file_path) + "\n"

        if cached_document_text and question:
            answer = answer_with_phi3(cached_document_text, question)
        else:
            answer = "Please upload PDF files and ask a question."

    return render_template("index.html", answer=answer)


if __name__ == "__main__":
    app.run(debug=True)
