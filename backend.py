import os
import requests
import chromadb
from dotenv import load_dotenv, find_dotenv

# Modern LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings  # Free Embeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv(find_dotenv(), override=True)

# Configuration
PERSIST_DIR = os.path.join(os.getcwd(), "chroma_db")
CHROMA_COLLECTION_NAME = "documents"

# Initialize Core Components (lazy initialization)
_llm = None
_embeddings = None

def get_llm():
    """Get or initialize the LLM instance (Using Groq)."""
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model_name=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE")
        )
    return _llm

def get_embeddings():
    """Get or initialize the embeddings instance (Using Free HuggingFace)."""
    global _embeddings
    if _embeddings is None:
        # Runs locally on your CPU - 100% Free
        _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embeddings

def get_vectorstore():
    """Returns the persistent Chroma vector store."""
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=get_embeddings(),
        collection_name=CHROMA_COLLECTION_NAME
    )

def ingest_document(uploaded_file):
    """
    Processes a PDF: Saves, loads, splits, and updates the vector DB.
    """
    temp_file_path = f"temp_{uploaded_file.name}"
    try:
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load and Split
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            add_start_index=True
        )
        splits = text_splitter.split_documents(docs)

        # Add to Chroma
        vectorstore = get_vectorstore()
        vectorstore.add_documents(documents=splits)
        
        return vectorstore

    except Exception as e:
        raise RuntimeError(f"Failed to ingest document: {str(e)}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def route_query(query: str) -> str:
    """
    Classifies intent: 'WEB' or 'DOC'.
    """
    router_template = """You are an expert routing agent.
    
    Decide whether to route the user's query to:
    - WEB: Current events, live data, comparisons, or general knowledge.
    - DOC: Specifics likely found in uploaded user documents.
    
    Query: {query}
    
    Return ONLY one word: "WEB" or "DOC".
    """
    
    try:
        prompt = PromptTemplate.from_template(router_template)
        chain = prompt | get_llm() | StrOutputParser()
        decision = chain.invoke({"query": query}).strip().upper()
        
        if "WEB" in decision: return "WEB"
        if "DOC" in decision: return "DOC"
        return "WEB" 
    except Exception:
        return "WEB"

def search_serper(query: str) -> str:
    """Fetches web results via Serper.dev."""
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return "Error: SERPER_API_KEY not found."

    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    
    try:
        response = requests.post(url, headers=headers, json={"q": query}, timeout=10)
        response.raise_for_status()
        results = response.json()
        
        snippets = []
        if "organic" in results:
            for item in results["organic"][:5]:
                snippets.append(f"- {item.get('title')}: {item.get('snippet')}")
        
        return "\n".join(snippets) if snippets else "No relevant web results found."
    except Exception as e:
        return f"Web search failed: {str(e)}"

def get_answer(query: str) -> str:
    """
    Orchestrates the RAG flow based on routing.
    """
    route = route_query(query)
    
    if route == "WEB":
        print(f"DEBUG: Routing '{query}' to WEB")
        context = search_serper(query)
        
        prompt = PromptTemplate.from_template(
            """Answer the question based ONLY on the following web search results.
            
            Search Results:
            {context}
            
            Question: {question}
            """
        )
        chain = prompt | get_llm() | StrOutputParser()
        response = chain.invoke({"context": context, "question": query})
        return f"**[Source: Web Search]**\n\n{response}"

    else:
        print(f"DEBUG: Routing '{query}' to DOC")
        vectorstore = get_vectorstore()
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        
        prompt = PromptTemplate.from_template(
            """Answer the question based ONLY on the following context. 
            If the answer is not in the context, strictly say "I cannot find this information in the document."
            
            Context:
            {context}
            
            Question: {question}
            """
        )
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | get_llm()
            | StrOutputParser()
        )
        
        try:
            response = rag_chain.invoke(query)
            return f"**[Source: Document]**\n\n{response}"
        except Exception as e:
            return f"Error querying documents: {str(e)}. (Did you upload a file?)"
