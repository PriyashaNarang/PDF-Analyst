import streamlit as st
import os
import time
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Analyst",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Remove Deploy button via JS (CSS selectors are unreliable across versions) ──
import streamlit.components.v1 as components
components.html("""
<script>
(function() {
  function hideDeployButton() {
    // Find every button and link in the parent window
    var els = window.parent.document.querySelectorAll('button, a');
    els.forEach(function(el) {
      var txt = el.innerText || el.textContent || '';
      var label = (el.getAttribute('aria-label') || '').toLowerCase();
      var title = (el.getAttribute('title') || '').toLowerCase();
      if (
        txt.trim() === 'Deploy' ||
        label.includes('deploy') ||
        title.includes('deploy')
      ) {
        el.style.setProperty('display', 'none', 'important');
      }
    });
  }
  // Run immediately and keep polling for 10s in case Streamlit renders it late
  hideDeployButton();
  var count = 0;
  var interval = setInterval(function() {
    hideDeployButton();
    count++;
    if (count > 20) clearInterval(interval);
  }, 500);
})();
</script>
""", height=0)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
/* Hide only the hamburger menu and footer — NOT the header, so sidebar toggle stays */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

.block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
.app-header {
    display: flex; align-items: center; gap: 12px;
    padding: 1rem 0 1.5rem; border-bottom: 1px solid #e5e7eb; margin-bottom: 1.5rem;
}
.app-header h1 { font-size: 1.4rem; font-weight: 600; color: #111827; margin: 0; }
.app-header span { font-size: 0.85rem; color: #6b7280; }
section[data-testid="stSidebar"] { background: #f9fafb; border-right: 1px solid #e5e7eb; }
section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
.pdf-card {
    background: #fff; border: 1px solid #e5e7eb; border-radius: 10px;
    padding: 0.875rem 1rem; margin-bottom: 0.75rem;
    display: flex; align-items: center; gap: 10px;
}
.pdf-card .icon { font-size: 1.3rem; }
.pdf-card .name { font-size: 0.85rem; font-weight: 500; color: #111827; overflow-wrap: anywhere; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 99px; font-size: 0.72rem; font-weight: 500; }
.badge-green { background: #dcfce7; color: #15803d; }
.source-chip {
    display: inline-block; font-size: 0.7rem; padding: 2px 8px; background: #eff6ff; color: #1d4ed8;
    border-radius: 99px; border: 1px solid #bfdbfe; font-family: 'JetBrains Mono', monospace;
    margin: 2px 3px 0 0;
}
.empty-state { text-align: center; padding: 3rem 1rem; color: #9ca3af; }
.empty-state .icon { font-size: 2.5rem; margin-bottom: 0.5rem; }
.stat-row { display: flex; gap: 8px; margin-top: 0.75rem; }
.stat-box { flex: 1; background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 0.6rem 0.75rem; text-align: center; }
.stat-box .val { font-size: 1.1rem; font-weight: 600; color: #111827; }
.stat-box .lbl { font-size: 0.68rem; color: #6b7280; margin-top: 1px; }
.stButton button { border-radius: 8px !important; font-weight: 500 !important; }
.stProgress > div > div { background: #1d4ed8 !important; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "messages": [],           # [{role, content, sources}]
        "chat_history": [],       # [HumanMessage, AIMessage, ...]
        "rag_chain": None,
        "pdf_name": None,
        "pdf_pages": 0,
        "pdf_chunks": 0,
        "ingested": False,
        "pending_question": None, # question queued by suggestion buttons
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── Core functions ────────────────────────────────────────────────────────────
def get_api_key():
    return os.getenv("GOOGLE_API_KEY", "")


@st.cache_resource(show_spinner=False)
def load_embeddings(api_key: str):
    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview",
        google_api_key=api_key,
    )


def safe_load_pdf(pdf_path: str):
    """Load PDF with PyMuPDF (handles corrupt pages, image pages, unusual structures)."""
    try:
        loader = PyMuPDFLoader(str(pdf_path))
        docs = loader.load()
        return [d for d in docs if d.page_content.strip()]
    except Exception:
        # Page-by-page fallback
        import fitz
        from langchain_core.documents import Document
        docs = []
        pdf = fitz.open(str(pdf_path))
        for i, page in enumerate(pdf):
            try:
                text = page.get_text("text").strip()
                if text:
                    docs.append(Document(
                        page_content=text,
                        metadata={"source": str(pdf_path), "page": i}
                    ))
            except Exception:
                pass
        pdf.close()
        return docs


def ingest_pdf(pdf_path: str, api_key: str, chunk_size: int, chunk_overlap: int):
    docs = safe_load_pdf(pdf_path)
    if not docs:
        raise ValueError("No readable text found. The PDF may be a scanned image.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    if not chunks:
        raise ValueError("PDF loaded but produced no text chunks.")

    embeddings = load_embeddings(api_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore, len(docs), len(chunks)


def build_rag_chain(vectorstore, api_key: str, top_k: int, temperature: float):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=temperature,
        google_api_key=api_key,
        convert_system_message_to_human=True,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the chat history and the latest user question, rewrite it as a "
         "standalone question. Do NOT answer — only rewrite if needed."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful PDF document analyst. Answer using ONLY the context below. "
         "If the answer is not in the context, say so clearly.\n\nContext:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    doc_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, doc_chain)


def format_sources(context_docs):
    seen, chips = set(), []
    for doc in context_docs:
        try:
            label = f"Page {int(doc.metadata.get('page', 0)) + 1}"
        except Exception:
            label = "Page ?"
        if label not in seen:
            seen.add(label)
            chips.append(label)
    return chips


def run_chain(question: str):
    """Call the RAG chain and append result to session messages."""
    try:
        result = st.session_state.rag_chain.invoke({
            "input": question,
            "chat_history": st.session_state.chat_history,
        })
        answer  = result.get("answer", "I couldn't find a relevant answer in the document.")
        sources = format_sources(result.get("context", []))
        st.session_state.chat_history.append(HumanMessage(content=question))
        st.session_state.chat_history.append(AIMessage(content=answer))
    except Exception as e:
        answer  = f"⚠️ Error calling Gemini: {e}"
        sources = []

    st.session_state.messages.append({
        "role": "assistant", "content": answer, "sources": sources
    })


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📄 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"], label_visibility="collapsed")

    with st.expander("Advanced settings", expanded=False):
        chunk_size    = st.slider("Chunk size (chars)",   500, 2000, 1000, 100)
        chunk_overlap = st.slider("Chunk overlap",          0,  400,  200,  50)
        top_k         = st.slider("Retrieved chunks (k)",   2,   10,    5,   1)
        temperature   = st.slider("Temperature",          0.0,  1.0,  0.2, 0.05)

    if uploaded_file:
        if not get_api_key():
            st.error("⚠️ GOOGLE_API_KEY not found. Add it to your .env file.")
        else:
            label = "🔄 Re-process PDF" if st.session_state.ingested else "🚀 Process PDF"
            if st.button(label, use_container_width=True, type="primary"):
                save_path = Path("uploads") / uploaded_file.name
                save_path.parent.mkdir(exist_ok=True)
                save_path.write_bytes(uploaded_file.read())
                prog = st.progress(0, text="Loading PDF pages…")
                try:
                    prog.progress(20, text="Reading & splitting PDF…")
                    vs, pages, chunks = ingest_pdf(str(save_path), get_api_key(), chunk_size, chunk_overlap)
                    prog.progress(70, text="Building RAG chain…")
                    chain = build_rag_chain(vs, get_api_key(), top_k, temperature)
                    prog.progress(100, text="Done!")
                    time.sleep(0.4)
                    prog.empty()

                    st.session_state.rag_chain        = chain
                    st.session_state.pdf_name         = uploaded_file.name
                    st.session_state.pdf_pages        = pages
                    st.session_state.pdf_chunks       = chunks
                    st.session_state.ingested         = True
                    st.session_state.messages         = []
                    st.session_state.chat_history     = []
                    st.session_state.pending_question = None
                    st.success("✅ PDF ready — start chatting!")
                    st.rerun()
                except Exception as e:
                    prog.empty()
                    st.error(f"Error: {e}")

    if st.session_state.ingested:
        st.divider()
        st.markdown(f"""
        <div class="pdf-card">
          <span class="icon">📄</span>
          <div>
            <div class="name">{st.session_state.pdf_name}</div>
            <div><span class="badge badge-green">✓ Ready</span></div>
          </div>
        </div>
        <div class="stat-row">
          <div class="stat-box"><div class="val">{st.session_state.pdf_pages}</div><div class="lbl">Pages</div></div>
          <div class="stat-box"><div class="val">{st.session_state.pdf_chunks}</div><div class="lbl">Chunks</div></div>
          <div class="stat-box"><div class="val">{len(st.session_state.messages)//2}</div><div class="lbl">Turns</div></div>
        </div>""", unsafe_allow_html=True)

        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.messages         = []
            st.session_state.chat_history     = []
            st.session_state.pending_question = None
            st.rerun()


# ── Main ──────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="app-header">
  <span style="font-size:1.8rem">📑</span>
  <div>
    <h1>PDF Analyst</h1>
    <span>Powered by LangChain · RAG · Gemini 2.5 Flash (free tier)</span>
  </div>
</div>
""", unsafe_allow_html=True)

SUGGESTIONS = [
    "Summarise the key points of this document",
    "What is the main argument or conclusion?",
    "List the most important findings or recommendations",
    "What methodology or approach is described?",
]

if not st.session_state.ingested:
    st.markdown("""
    <div class="empty-state">
      <div class="icon">⬆️</div>
      <p><strong>Upload a PDF in the sidebar to get started.</strong><br>
      Ask anything about the document — summaries, details, comparisons, and more.</p>
    </div>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1: st.markdown("**1. Upload**\nDrop any PDF in the sidebar.")
    with col2: st.markdown("**2. Process**\nDoc is chunked, embedded, and indexed.")
    with col3: st.markdown("**3. Chat**\nAsk questions; Gemini answers from the doc.")

else:
    # ── Step 1: If a suggestion was clicked last run, call the chain NOW ──────
    # Streamlit buttons cannot call the chain directly (they trigger a rerun
    # before we can call invoke). So buttons write to pending_question, and
    # on the NEXT run (here) we detect it, call the chain, clear the flag,
    # and rerun once more to display everything cleanly.
    if st.session_state.get("pending_question"):
        question = st.session_state.pending_question
        st.session_state.pending_question = None
        with st.spinner("Thinking…"):
            run_chain(question)
        st.rerun()

    # ── Step 2: Show suggestion buttons only when no chat yet ─────────────────
    if not st.session_state.messages:
        st.markdown("#### Suggested questions")
        cols = st.columns(2)
        for i, q in enumerate(SUGGESTIONS):
            with cols[i % 2]:
                if st.button(q, key=f"sugg_{i}", use_container_width=True):
                    # Queue the user message + the question, then rerun
                    st.session_state.messages.append({"role": "user", "content": q})
                    st.session_state.pending_question = q
                    st.rerun()

    # ── Step 3: Render existing messages ─────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                chips = "".join(f'<span class="source-chip">{s}</span>' for s in msg["sources"])
                st.markdown(f'📎 {chips}', unsafe_allow_html=True)

    # ── Step 4: Chat input ────────────────────────────────────────────────────
    user_input = st.chat_input("Ask anything about your PDF…")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                run_chain(user_input)
            last = st.session_state.messages[-1]
            st.markdown(last["content"])
            if last.get("sources"):
                chips = "".join(f'<span class="source-chip">{s}</span>' for s in last["sources"])
                st.markdown(f'📎 {chips}', unsafe_allow_html=True)
        st.rerun()