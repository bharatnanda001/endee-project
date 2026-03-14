"""
ui.py — Streamlit Chat Interface for Endee RAG Research Assistant

Run:
    streamlit run ui.py

Requirements:
- pip install streamlit
- API must be running: uvicorn app:app --reload
"""

import streamlit as st
import requests

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Endee RAG Research Assistant",
    page_icon="🔍",
    layout="wide",
)

API_BASE = "http://localhost:8000"

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://endee.io/favicon.ico", width=40)
    st.title("⚙️ Settings")
    top_k = st.slider("Top-K Results", min_value=1, max_value=5, value=3)
    show_sources = st.checkbox("Show retrieved documents", value=True)
    st.markdown("---")
    st.markdown("**Powered by:**")
    st.markdown("🗄️ Endee Vector Database")
    st.markdown("🤗 Sentence Transformers")
    st.markdown("⚡ FastAPI + Streamlit")
    st.markdown("---")
    st.markdown("[GitHub](https://github.com/endee-io/endee) | [Docs](https://docs.endee.io)")

# ─── Header ───────────────────────────────────────────────────────────────────
st.title("🔍 Endee RAG Research Assistant")
st.markdown(
    "Ask questions about AI, vector databases, RAG, and more. "
    "Answers are grounded by **semantic retrieval** from the **Endee vector database**."
)
st.markdown("---")

# ─── Chat History ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ─── Input ────────────────────────────────────────────────────────────────────
query = st.chat_input("Ask a question (e.g. 'What is RAG?' or 'How does Endee work?')")

if query:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Call API
    with st.chat_message("assistant"):
        with st.spinner("Searching Endee & generating answer..."):
            try:
                response = requests.get(
                    f"{API_BASE}/chat",
                    params={"q": query, "top_k": top_k},
                    timeout=30,
                )
                data = response.json().get("response", {})
                answer = data.get("answer", "No answer returned.")
                docs = data.get("retrieved_documents", [])

                st.markdown(answer)

                if show_sources and docs:
                    st.markdown("---")
                    st.markdown("**📚 Retrieved Documents:**")
                    for i, doc in enumerate(docs, 1):
                        similarity_pct = f"{doc['similarity'] * 100:.1f}%"
                        with st.expander(f"📄 Doc {i} — Similarity: {similarity_pct} — Source: {doc['source']}"):
                            st.markdown(doc["text"])

                st.session_state.messages.append({"role": "assistant", "content": answer})

            except requests.exceptions.ConnectionError:
                err = "❌ Cannot connect to the API. Make sure to run: `uvicorn app:app --reload`"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
            except Exception as e:
                err = f"❌ Error: {str(e)}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
