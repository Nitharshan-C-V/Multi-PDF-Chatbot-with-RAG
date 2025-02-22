
import bs4
import os 
from dotenv import load_dotenv
from langchain import hub

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings



#LANGSMITH
#os.environ['LANGCHAIN_TRACING_V2'] = 'true'
#os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
#os.environ['LANGCHAIN_PROJECT']='advanced_rag'
#os.environ['LANGCHAIN_API_KEY']

#os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

load_dotenv()
#### INDEXING ####

# Load Documents
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

#print(docs)


##Split -Text Chunking 

text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000 , chunk_overlap=200)
splits= text_splitter.split_documents(docs)

##Embedding 

model_name="BAAI/bge-small-en"
model_kwargs={'device':'cpu'}
encode_kwargs= {"normalized_embeddings" : True}
hf_embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

print(hf_embeddings)

vectorstore= FAISS.from_documents(documents=splits,embedding=hf_embeddings)

retriever=vectorstore.as_retriever() #Dense Retrival - Embeddings/Context based

##Prommpt
prompt= hub.pull("rlm/rag-prompt")

#LLM
llm= ChatGroq(model="llama3-8b-8192",temperature=0)

#Post Processing

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Question
print(rag_chain.invoke("What is Task Decomposition?"))