import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
import os
import time
from openai import RateLimitError

# --- SIDEBAR: Projektval och knowledge base ---
st.sidebar.title("Upphandlingsprojekt")
project_name = st.sidebar.text_input("Namn på nytt projekt", "upphandling-1")
create_project = st.sidebar.button("Skapa nytt projekt")

project_path_root = "project-data/projects"
os.makedirs(project_path_root, exist_ok=True)

if create_project:
    os.makedirs(f"{project_path_root}/{project_name}/uploaded_files", exist_ok=True)
    st.sidebar.success(f"Projekt '{project_name}' skapat!")

selected_project = st.sidebar.selectbox("Välj aktivt projekt", os.listdir(project_path_root))

# --- Knowledge Base ---
@st.cache_resource
def load_knowledge_base():
    kb_path = "project-data/customer-info"
    all_docs = []
    for file in os.listdir(kb_path):
        if not file.endswith(".pdf"):
            continue
        file_path = os.path.join(kb_path, file)
        loader = PyMuPDFLoader(file_path)
        all_docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(all_docs)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

kb_vectorstore = load_knowledge_base()

# --- Filuppladdning ---
st.title("AI-verktyg för offentliga upphandlingar")
st.header(f"Projekt: {selected_project}")

uploaded_files = st.file_uploader("Ladda upp upphandlingsfiler (PDF)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    project_path = f"{project_path_root}/{selected_project}/uploaded_files"
    os.makedirs(project_path, exist_ok=True)
    for file in uploaded_files:
        with open(os.path.join(project_path, file.name), "wb") as f:
            f.write(file.getbuffer())
    st.success("Filer uppladdade!")

# --- Generera struktur över upphandlingen ---
if st.button("🔍 Skapa upphandlingsstruktur"):
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.2, max_tokens=2048)
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=kb_vectorstore.as_retriever(), chain_type="stuff")

    structure_prompt = "Analysera följande text och identifiera alla delar som kräver fritextsvar i upphandlingen. Lista dem med rubriker."
    summaries = []

    for file in os.listdir(f"{project_path_root}/{selected_project}/uploaded_files"):
        if not file.endswith(".pdf"):
            continue

        file_path = os.path.join(f"{project_path_root}/{selected_project}/uploaded_files", file)
        loader = PyMuPDFLoader(file_path)
        pages = loader.load()
        content = "\n".join([p.page_content for p in pages])

        try:
            result = chain.run(f"{structure_prompt}\n\n{content}")
        except RateLimitError:
            st.warning("Rate limit från OpenAI – väntar 20 sekunder...")
            time.sleep(20)
            result = chain.run(f"{structure_prompt}\n\n{content}")

        summaries.append(f"### {file}\n{result}")

    st.subheader("📋 Föreslagen struktur")
    st.markdown("\n\n".join(summaries))

# --- Fråga AI:n ---
st.markdown("---")
st.subheader("🤖 Generera svar på upphandlingsfråga")
user_query = st.text_area("Ställ din fråga eller klistra in en rubrik du vill svara på:")

if st.button("✍️ Skriv svar") and user_query:
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.4, max_tokens=2048)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=kb_vectorstore.as_retriever())
    try:
        answer = qa_chain.run(user_query)
    except RateLimitError:
        st.warning("Rate limit från OpenAI – väntar 20 sekunder...")
        time.sleep(20)
        answer = qa_chain.run(user_query)
    st.text_area("Förslag på svar:", value=answer, height=300)

# --- Footer ---
st.markdown("---")
st.markdown("Byggd med ❤️ av Oscar och AI. MVP-version.")
