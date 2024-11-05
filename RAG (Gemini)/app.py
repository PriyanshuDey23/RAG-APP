import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Streamlit application title
st.title("RAG Application using Gemini Pro")

# Load PDF document
loader = PyPDFLoader("transformer.pdf")
data = loader.load() 
# pdf_path = "E:\\RAG (Gemini)\\transformer.pdf"
# loader = PyPDFLoader(pdf_path)
# data = loader.load()

# Split the document into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(data)

# Create embeddings and vectorstore
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma.from_documents(
    docs, 
    embedding=embedding_model,
    persist_directory="./chroma_data"  # Ensure the directory exists and is accessible
)

# Create a retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# Define the LLM and the system prompt
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-002", temperature=0,max_tokens=None,timeout=None)


# Streamlit input
query = st.chat_input("Ask me anything: ")


system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer in bullet format."
    "\n\n{context}"
)

# Prompt template for the LLM
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)



# If there's a query, process it
if query:
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": query})
    print(response["answer"])

    st.write(response["answer"])
