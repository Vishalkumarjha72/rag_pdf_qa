from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Typed application config, loaded from environment variables / .env file.
    Using pydantic-settings instead of raw os.getenv() calls so that:
      - missing required vars fail loudly at startup, not mid-request
      - values are type-checked (e.g. PINECONE_DIMENSION is guaranteed an int)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # V4: .env now also carries LANGSMITH_* vars, which LangSmith reads
        # directly from os.environ (via main.py's load_dotenv() call) rather
        # than through this Settings object — they're not declared as fields
        # here on purpose. Without extra="ignore", pydantic-settings' default
        # behavior is to ERROR on any .env variable it doesn't recognize as a
        # declared field, which is what broke startup.
        extra="ignore",
    )

    # OpenAI
    openai_api_key: str

    # Pinecone
    pinecone_api_key: str
    pinecone_index_name: str = "rag-index"
    pinecone_cloud: str = "aws"
    pinecone_environment: str = "us-east-1"

    # Embedding model — bge-base-en-v1.5 outputs 768-dim vectors
    embedding_model_name: str = "BAAI/bge-base-en-v1.5"
    embedding_dimension: int = 768

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379


# Single shared settings instance, imported wherever config is needed
settings = Settings()