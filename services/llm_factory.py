import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

load_dotenv()


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"환경변수 {name} 가 설정되지 않았습니다.")
    return value


def get_chat_llm() -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_endpoint=require_env("AOAI_ENDPOINT"),
        api_key=require_env("AOAI_API_KEY"),
        api_version=os.getenv("AOAI_API_VERSION", "2024-02-01"),
        azure_deployment=os.getenv("AOAI_DEPLOY_GPT4O_MINI", "gpt-4o-mini"),
        temperature=0.2,
    )


def get_embeddings() -> AzureOpenAIEmbeddings:
    return AzureOpenAIEmbeddings(
        azure_endpoint=require_env("AOAI_ENDPOINT"),
        api_key=require_env("AOAI_API_KEY"),
        api_version=os.getenv("AOAI_API_VERSION", "2024-02-01"),
        azure_deployment=os.getenv("AOAI_DEPLOY_EMBED_3_SMALL", "text-embedding-3-small"),
    )