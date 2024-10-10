from langchain_chroma import Chroma  # Chroma for vector store
from langchain_openai import ChatOpenAI  # OpenAI model for generation
from langchain.prompts import ChatPromptTemplate

from prompt_toolkit import HTML, PromptSession
from prompt_toolkit.history import FileHistory
from get_embedding_function import get_embedding_function
from log_init import logger

# Define constants
CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

_TEXT_COLOR_MAPPING = {
    "blue": "36;1",
    "yellow": "33;1",
    "pink": "38;5;200",
    "green": "32;1",
    "red": "31;1",
}


def get_colored_text(text: str, color: str) -> str:
    """Get colored text for terminal display."""
    color_str = _TEXT_COLOR_MAPPING[color]
    return f"\u001b[{color_str}m\033[1;3m{text}\u001b[0m"


def rag_executor(query: str) -> str:
    """Function to execute the RAG pipeline."""

    # Prepare the DB and model
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB for relevant documents
    results = db.similarity_search_with_score(query, k=5)

    # Create the context from retrieved documents
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # Prepare the prompt using context and the original query
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query)

    # Generate the response using the OpenAI model
    model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    response_text = model.invoke(prompt)

    # Get metadata (e.g., source IDs) for the retrieved documents
    sources = [doc.metadata.get("id", None) for doc, _score in results]

    # Return the response and source information
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    return formatted_response


if __name__ == "__main__":
    # Use the agent executor to handle the RAG process
    session = PromptSession(history=FileHistory(".rag-history-file"))
    while True:
        # Prompt user for query
        question = session.prompt(
            HTML("<b>Type <u>Your question</u></b>  ('q' to exit): ")
        )
        if question.lower() in ["q", "exit"]:
            break
        if len(question) == 0:
            continue
        try:
            # Execute the RAG pipeline and print results
            response = rag_executor(question)
            logger.info(get_colored_text(response, "green"))
        except Exception as e:
            logger.info(
                get_colored_text("Error occurred in agent", "red"),
                get_colored_text(str(e), "red"),
            )
