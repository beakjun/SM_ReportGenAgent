from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langsmith import Client


load_dotenv('.env')

# llm
agent_llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0)
chat_llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

# 랭스미스
client = Client()

# DB
db = SQLDatabase.from_uri('postgresql://postgres:postgres@10.10.50.155:1108/postgres', schema="doosan")

# 벡터 DB
embedding_model_name = "nlpai-lab/KoE5"
embedding = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs={
        "device": "cpu",
        # "trust_remote_code": True,  # 모델에 따라 필요할 수 있음
    },
    # encode_kwargs={"normalize_embeddings": True},  # 코사인 유사도 안정화 (버전에 따라 지원)
)

vectorstore = Chroma(
    persist_directory='./chroma_reports_db',
    embedding_function=embedding,
    collection_name="reports_ko1" 
)