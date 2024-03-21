from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())



loader = PyPDFLoader("Cocina y nutrición.pdf")
pages = loader.load_and_split()

# Inicializamos creador de vectores
embeddings = OpenAIEmbeddings()
# Cargamos los vectores
vector_store = FAISS.from_documents(pages, embeddings)
# Creamos un recuperador de vectores
retriever = vector_store.as_retriever()
# Cargamos el prompt desde el hub de langchain
prompt = hub.pull("rlm/rag-prompt")
# Inicializamos el modelo de lenguaje
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
# Creamos la cadena de chat
chain = {
    "context": retriever, 
    "question": RunnablePassthrough()
    } | prompt | llm | StrOutputParser()