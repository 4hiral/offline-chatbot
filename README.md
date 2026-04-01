# Offline Document Chatbot — Setup Guide

## Folder Structure

```
offline_chatbot/
├── app.py
├── requirements.txt
├── README.md
└── templates/
    └── index.html
```

---

## Step 1 — Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 2 — Install Poppler (required for PDF rendering)

1. Download from: https://github.com/oschwartz10612/poppler-windows/releases
2. Extract it (e.g. to `C:\poppler\`)
3. Set the env var OR edit `POPPLER_PATH` in `app.py`:
   ```
   set POPPLER_PATH=C:\poppler\Library\bin
   ```

---

## Step 3 — Install Tesseract (OCR fallback for low-quality scans)

1. Download from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install to default path: `C:\Program Files\Tesseract-OCR\`
3. If installed elsewhere, set env var:
   ```
   set TESSERACT_PATH=C:\your\path\tesseract.exe
   ```

---

## Step 4 — Start LM Studio

1. Open LM Studio
2. Download model: `meta-llama-3-8b-instruct` (or any GGUF model)
3. Go to **Developer → Local Server**
4. Set Status to **Running**
5. Default URL: `http://127.0.0.1:1234`

---

## Step 5 — Run the App

```bash
cd offline_chatbot
python app.py
```

Open browser: http://localhost:5000

---

## Environment Variables (all optional)

| Variable            | Default                                      | Description                          |
|---------------------|----------------------------------------------|--------------------------------------|
| `POPPLER_PATH`      | `C:\...\poppler-25.12.0\Library\bin`         | Path to Poppler bin folder           |
| `TESSERACT_PATH`    | `C:\Program Files\Tesseract-OCR\tesseract.exe` | Path to tesseract.exe              |
| `LM_BASE_URL`       | `http://127.0.0.1:1234/v1`                   | LM Studio server URL                 |
| `LLM_MODEL`         | `meta-llama-3-8b-instruct`                   | Model name shown in LM Studio        |
| `OCR_DPI`           | `200`                                        | DPI for scanned page rendering       |
| `TABLE_DPI`         | `300`                                        | DPI for table extraction             |
| `MAX_UPLOAD_MB`     | `500`                                        | Max upload file size in MB           |
| `PADDLE_CONF_FLOOR` | `0.75`                                       | Confidence below which Tesseract runs|

---

## How It Works

### Document Upload
- Upload PDF, DOCX, PNG, or JPG files
- Each page is checked individually:
  - **Digital page** (has text layer) → text extracted directly via pdfplumber (fast, accurate)
  - **Scanned page** (no text layer) → converted to image → preprocessed → OCR via PaddleOCR + Tesseract fallback
- Text is chunked and indexed in a FAISS vector store for Q&A

### Q&A
- Your question is embedded and matched against indexed chunks
- Top matching chunks are sent to the LLM as context
- LLM answers strictly from the context (no hallucination)

### Table Extraction
- Click **Extract Tables** or type "show table" / "table on page 3"
- Tries text-layer extraction first (fast)
- Falls back to PPStructure (deep learning table detection) for scanned tables
- Tables displayed in chat and automatically saved to `uploads/extracted_tables.xlsx`

---

## Recommended Models (if llama-3-8b gives poor results)

| Model | Why |
|-------|-----|
| `Mistral-7B-Instruct-v0.3` | Better instruction following, less hallucination |
| `Phi-3-mini-4k-instruct`   | Faster, lower RAM, good for extraction tasks     |
| `Qwen2.5-7B-Instruct`      | Excellent at structured output / tables          |

To switch: change `LLM_MODEL` env var or edit `app.py` line 34.
