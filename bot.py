from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
#from langchain_core import documents
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
#import openai
from langchain.prompts.prompt import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma
import streamlit as st
import tempfile


def load_document(file):

    try:
        file_extension = file.type.split('/')[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
            tmp_file.write(file.read())
            tmp_file_path = tmp_file.name

        if file.type == "application/pdf":
            loader = PyPDFLoader(tmp_file_path)
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" or file.type == "application/msword":
            loader = Docx2txtLoader(tmp_file_path)
        elif file.type == "text/plain":
            loader = TextLoader(tmp_file_path)
        else:
            raise ValueError("Unsupported file format. Please upload a PDF, DOC/DOCX, or TXT file.")

        return loader.load()

    finally:
        if tmp_file_path:
            os.remove(tmp_file_path)


def create_chatbot(doc):

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(doc)

    openai_api_key = os.environ.get("openai_api_key")
    llm_model = 'text-embedding-ada-002'
    embeddings = OpenAIEmbeddings(model=llm_model, api_key=openai_api_key)

    vector = Chroma.from_documents(chunks, embeddings)
    retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    custom_template = """Given the chat history and a follow-up question, rephrase the follow-up question as a standalone question that doesn’t rely on prior context.

    Chat History:

    {chat_history}

    Follow Up Input: {question}
    Standalone question:"""

    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)
    Model = 'gpt-3.5-turbo'

    qa = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(temperature=0, model=Model, api_key=openai_api_key, max_tokens=800),
        retriever,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        return_source_documents=True
    )
    return qa


st.title("Lang-Chain ChatBot")

# three type of file can be uploaded : pdf, doc, text
uploaded_file = st.file_uploader("Upload a PDF, DOC/DOCX, or TXT file", type=['pdf', 'docx', 'doc', 'txt'])

# initialize session if not initialized
if 'qa' not in st.session_state:
    st.session_state.qa = None
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None

# handle file upload
if uploaded_file is not None:

    # if a new file is uploaded, reset the chatbot
    if st.session_state.qa is None or st.session_state.uploaded_file_name != uploaded_file.name:
        st.session_state.qa = create_chatbot(load_document(uploaded_file))
        st.session_state.uploaded_file_name = uploaded_file.name

    qa = st.session_state.qa

    # input for user's question
    query = st.text_input("Write your queries:")

    # submit user's question
    if st.session_state.uploaded_file_name == uploaded_file.name:
        if query:
            result = qa({"question": query, "chat_history": []})
            answer = result["answer"]


            st.write(f"**Bot:** {answer}")
else:
    st.write("Please upload a document to start the chat.")