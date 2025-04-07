import streamlit as st
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os
import tempfile

# --- SIDEBAR: Projektval och knowledge base ---
st.sidebar.title("Upphandlingsprojekt")
project_name = st.sidebar.text_input("Namn på nytt projekt", "upphandling-1")
create_project = st.sidebar.button("Skapa nytt projekt")

if create_project:
    os.makedirs(f"project-data/projects/{project_name}/uploaded_files", exist_ok=True)
    st.sidebar.success(f"Projekt '{project_name}' skapat!")

selected_project = st.sidebar.selectbox("Välj aktivt projekt", os.listdir("project-data/projects"))

# --- Knowledge Base (engångsladdning) ---
@st.cache_resource
def load_knowledge_base():
    kb_path = "project-data/customer-info"
    all_docs = []
    for file in os.listdir(kb_path):
        if file.endswith(".pdf"):
            loader = UnstructuredPDFLoader(os.path.join(kb_path, file))
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

uploaded_files = st.file_uploader("Ladda upp upphandlingsfiler (PDF)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    project_path = f"project-data/projects/{selected_project}/uploaded_files"
    os.makedirs(project_path, exist_ok=True)
    for file in uploaded_files:
        with open(os.path.join(project_path, file.name), "wb") as f:
            f.write(file.getbuffer())
    st.success("Filer uppladdade!")

# --- Generera struktur över upphandlingen ---
if st.button("🔍 Skapa upphandlingsstruktur"):
    full_text = ""
    for file in os.listdir(f"project-data/projects/{selected_project}/uploaded_files"):
        loader = UnstructuredPDFLoader(os.path.join(f"project-data/projects/{selected_project}/uploaded_files", file))
        pages = loader.load()
        full_text += "\n".join([page.page_content for page in pages])

    llm = ChatOpenAI(model_name="gpt-4", temperature=0.2)
    structure_prompt = "Analysera följande text och identifiera alla delar som kräver fritextsvar i upphandlingen. Lista dem med rubriker."
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=kb_vectorstore.as_retriever(), chain_type="stuff")
    result = chain.run(f"{structure_prompt}\n\n{full_text}")
    st.subheader("📋 Föreslagen struktur")
    st.write(result)

# --- Fråga AI:n ---
st.markdown("---")
st.subheader("🤖 Generera svar på upphandlingsfråga")
user_query = st.text_area("Ställ din fråga eller klistra in en rubrik du vill svara på:")

if st.button("✍️ Skriv svar") and user_query:
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.4)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=kb_vectorstore.as_retriever())
    answer = qa_chain.run(user_query)
    st.text_area("Förslag på svar:", value=answer, height=300)

# --- Footer ---
st.markdown("---")
st.markdown("Byggd med ❤️ av Oscar och AI. MVP-version.")
