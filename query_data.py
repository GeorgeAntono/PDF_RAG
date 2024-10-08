import argparse
from langchain_chroma import Chroma  # Updated import from langchain-chroma package
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    # Extract context from search results
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # Prepare the prompt using context and the original query
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Initialize the OpenAI chat model
    model = ChatOpenAI(temperature=0,model="gpt-3.5-turbo")  # Adjust model settings as needed
    response_text = model.invoke(prompt)

    # Get document metadata (e.g., source IDs)
    sources = [doc.metadata.get("id", None) for doc, _score in results]

    # Format and print the response along with sources
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()
