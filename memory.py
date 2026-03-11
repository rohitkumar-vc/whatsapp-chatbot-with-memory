from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_neo4j import Neo4jVector
from mem0 import Memory

from config import app_settings


langchain_embeddings = GoogleGenerativeAIEmbeddings(
    model=app_settings.EMBEDDING_MODEL,
    api_key=app_settings.GOOGLE_API_KEY,
    vertexai=app_settings.VERTEX_AI,
)

vector_store = Neo4jVector(
    embedding=langchain_embeddings,
    url=app_settings.NEO4J_URL,
    username=app_settings.NEO4J_USERNAME,
    password=app_settings.NEO4J_PASSWORD,
    index_name="mem0",  # Tells Neo4j to name the search index "mem0"
    node_label="mem0",  # Tags all memory nodes with the label "mem0"
    text_node_property="text",  # Tells Neo4j which field holds the actual memory text
)

# llm initialization

gemini_model = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    api_key=app_settings.GOOGLE_API_KEY,
    vertexai=app_settings.VERTEX_AI,
)

config = {
     "vector_store": {
        "provider": "langchain",
        "config": {
            "client": vector_store
        }
    },
    "llm": {
        "provider": "langchain",
        "config": {
            "model": gemini_model,
        },
    },
    "embedder": {
        "provider": "langchain",
        "config": {
            "model": langchain_embeddings,
        },
    },
    "reranker": {
        "provider": "cohere",
        "config": {"model": "rerank-english-v3.0"},
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": app_settings.NEO4J_URL,
            "username": app_settings.NEO4J_USERNAME,
            "password": app_settings.NEO4J_PASSWORD,
            "database": "neo4j",
        },
    },
}
