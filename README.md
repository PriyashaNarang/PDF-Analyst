# 📑 PDF Analyst — Conversational RAG with LangChain + Gemini

A Streamlit web app that lets you upload any PDF and have a full conversation
about it — powered by LangChain, FAISS vector search, and Google Gemini 2.5.

---

## Project structure

```
pdf-analyst/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── .streamlit/
│   └── config.toml         # Streamlit theme config
├── uploads/                # Saved PDFs (auto-created)
└── vectorstore/            # Persisted FAISS indexes (auto-created)
```

---

## Prerequisites

- Python 3.9 or newer
- A Google API key with access to Gemini and the Embedding API

### Get a Google API key

1. Go to https://makersuite.google.com/app/apikey
2. Sign in with your Google account
3. Click **Create API key**
4. Copy the key — you will use it in Step 3 below

---

## Setup steps

### Step 1 — Clone or download the project

If you have git:
```bash
git clone <your-repo-url>
cd pdf-analyst
```

Or just place all the files in a folder called `pdf-analyst` and `cd` into it.

---

### Step 2 — Create a virtual environment

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

You should see `(venv)` at the start of your terminal prompt.

---

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

This installs Streamlit, LangChain, FAISS, PyPDF, and the Gemini SDK.
It may take 1–2 minutes.

---

### Step 4 — Set your API key

**Option A — .env file (recommended):**
```bash
cp .env.example .env
```
Open `.env` in any text editor and replace `your_google_api_key_here` with
your actual key:
```
GOOGLE_API_KEY=AIzaSyABC123...
```

**Option B — Environment variable:**
```bash
# macOS / Linux
export GOOGLE_API_KEY=AIzaSyABC123...

# Windows Command Prompt
set GOOGLE_API_KEY=AIzaSyABC123...
```

**Option C — Enter it in the app UI.**
You can type the key directly into the sidebar after the app opens.
This is fine for quick testing.

---

### Step 5 — Run the app

```bash
streamlit run app.py
```

The terminal will show:
```
  You can now view your Streamlit app in your browser.

  Local URL:  http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Open http://localhost:8501 in your browser.

---

## How to use the app

1. **Enter your API key** in the sidebar (if you did not use a .env file).
2. **Upload a PDF** using the file uploader in the sidebar.
3. **Click "Process PDF"** — this embeds the document and builds the index.
   - Typical time: 5–30 seconds depending on PDF length.
4. **Ask questions** in the chat box at the bottom.
   - Click any suggested question to get started quickly.
   - Each answer shows which page(s) the information came from.
5. **Adjust settings** in the Advanced settings panel if needed.

---

## Advanced settings explained

| Setting | Default | Effect |
|---|---|---|
| Chunk size | 1000 | Characters per text chunk. Larger = more context per chunk, but fewer retrieved. |
| Chunk overlap | 200 | Characters shared between adjacent chunks. Prevents answers being cut at boundaries. |
| Retrieved chunks (k) | 5 | Number of chunks passed to Gemini as context. Higher = more coverage, slower. |
| Response temperature | 0.2 | 0 = factual/deterministic, 1 = more creative. Keep low for document Q&A. |

---

## Troubleshooting

**"google.api_core.exceptions.InvalidArgument"**  
Your API key may not have the Generative Language API enabled. Go to
https://console.cloud.google.com/apis/library, search for
"Generative Language API", and enable it.

**"ModuleNotFoundError: No module named 'faiss'"**  
Run: `pip install faiss-cpu`

**Slow on first question**  
The first query builds the retriever index in memory. Subsequent queries are faster.

**PDF not loading**  
Ensure the PDF is not password-protected and is a standard PDF (not a scanned image
without OCR). For scanned PDFs, you would need to add an OCR step.

---

## Tech stack

| Component | Library | Purpose |
|---|---|---|
| UI | Streamlit | Web interface |
| PDF loading | PyPDF / LangChain | Extract text from PDF |
| Chunking | RecursiveCharacterTextSplitter | Split text intelligently |
| Embeddings | Google `embedding-001` | Convert text to vectors |
| Vector store | FAISS | Fast similarity search |
| LLM | Gemini 1.5 Flash | Answer generation |
| Memory | ConversationBufferMemory | Chat history across turns |
| Orchestration | LangChain ConversationalRetrievalChain | Wire everything together |