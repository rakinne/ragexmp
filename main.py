import os
from langchain_anthropic import ChatAnthropic
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore


os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")

chat = ChatAnthropic(model="claude-sonnet-4-6")
embeddings = OllamaEmbeddings(model="llama3")
vectorstore = InMemoryVectorStore(embeddings=embeddings)

def check_anthropic_api_key():
    try:
        response = chat.invoke("What is the capital of France?")
        print(f"Anthropic API key is working. Response: {response.content}")
    except Exception as e:
        print(f"Anthropic API key check failed: {e}")

if __name__ == "__main__":
    check_anthropic_api_key()