 Universal Document Intelligence Chatbot

This project is a local Streamlit app that ingests PDFs, stores embeddings in ChromaDB, and answers queries either from uploaded documents or via web search.

Quick start (PowerShell):

```powershell
mkdir Universal_Doc_Chatbot
cd Universal_Doc_Chatbot
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
# Edit .env and add your keys
streamlit run app.py
```

Files created:
- `requirements.txt` - dependency list
- `backend.py` - ingestion, routing, and answer logic
- `app.py` - Streamlit UI
- `.env.example` - example API variables
- `README.md` - this file
- `.gitignore` - ignore venv, chroma_db, etc.

Notes & troubleshooting:
- If Chroma import errors occur, try installing the specific `chromadb` version compatible with your LangChain version.
- If OpenAI errors occur, ensure `OPENAI_API_KEY` is set in `.env`.
- The code uses `GPT-4o` via `model_name="gpt-4o"`; replace with `gpt-3.5-turbo` for cheaper testing.
