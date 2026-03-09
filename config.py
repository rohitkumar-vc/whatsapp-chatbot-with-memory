from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

class Settings(BaseSettings):
    
    # neo4j
    NEO4J_URL: str
    NEO4J_USERNAME: str
    NEO4J_PASSWORD: str
    
    # vectexAI
    GOOGLE_API_KEY:str
    VERTEX_AI: bool = True
    EMBEDDING_MODEL: str = "models/gemini-embedding-001"
    
    # rerank
    COHERE_API_KEY:str
    
    
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
app_settings = Settings()