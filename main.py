import os
import bs4
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_anthropic import ChatAnthropic
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

def check_anthropic_api_key():
    try:
        response = chat.invoke("Where is LeBron James and the Cleveland Cavaliers celebrating their 10yr championship anniversary?")
        print(f"Anthropic API key is working. Response: {response.content}")
    except Exception as e:
        print(f"Anthropic API key check failed: {e}")


# Below is a minimal helper for demonstration purposes.
def load_web_page(url: str, bs_kwargs: dict | None = None) -> list[Document]:
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    soup = bs4.BeautifulSoup(response.text, "html.parser", **(bs_kwargs or {}))
    return [Document(page_content=soup.get_text(), metadata={"source": url})]

if __name__ == "__main__":
    os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")

    chat = ChatAnthropic(model="claude-sonnet-4-6")
    embeddings = OllamaEmbeddings(model="llama3")
    vectorstore = InMemoryVectorStore(embedding=embeddings)

    # check_anthropic_api_key()
    # Only keep post title, headers, and content from the full HTML.
    bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
    docs = load_web_page(
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        bs_kwargs={"parse_only": bs4_strainer},
    )

    assert len(docs) == 1
    print(f"Total characters: {len(docs[0].page_content)}")
    print(docs[0].page_content[:500])

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # chunk size (characters)
        chunk_overlap=200,  # chunk overlap (characters)
        add_start_index=True,  # track index in original document
    )   
    all_splits = text_splitter.split_documents(docs)
    print(f"Split blog post into {len(all_splits)} sub-documents.")

    # embed the splits
    vectorstore.add_documents(all_splits)
    
    