# RAG with bedrock AWS

A simple RAG LangChain Implementation


# Import library/package
from langchain.embeddings import BedrockEmbeddings
from langchain.indexes import VectorStoreIndexCreator
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loader import PyPDFLoader
from langchain.llms.bedrock import Bedrock




# Load document
loader = PyPDFLoader(file_path="~/docs/shareholder-letter.pdf")


# Spilt doc into smaller chunk before embedding

text_splitter = RecursiveCharacterTextSplitter(
  seperators = ["\n\n", "\n", ".", " "],
  chunk_size=1000,
  chunk_overlap=100
)


# Create Embedding client using bedrock Titan Embedding Model
embeddings = BedrockEmbeddings(
  credentials_profile_name=os.environ.get("..."),
  region_name=os.environ.get("..."),
  endpoint_url=os.environ.get("...")
)


# Create vector store, embeddings and derive the an index for the document
index_creator = VectorStoreIndexCreator(
  vectorestore_cls=FAISS,
  embeddings=embeddings,
  text_splitter=text_splitter
)
index = index_creator.from_loaders([loader])

# Create the main text LLM for reasoning summarization QA
llm_bedrock=Bedrock(...)


# Invoke LLM -Search Query against Index, provide that as context to LLM
response = index.query(
  question = "What was the overall profit in 2022",
  llm=llm_bedrock
)
print(response)
