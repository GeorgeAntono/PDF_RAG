import chainlit as cl
from literalai import LiteralClient
from langchain_chroma import Chroma  # Chroma for vector store
from langchain_openai import ChatOpenAI  # OpenAI model for generation
from langchain.prompts import ChatPromptTemplate
from get_embedding_function import get_embedding_function
import os
literal_client = LiteralClient(api_key=os.getenv("LITERAL_API_KEY"))


# Define constants
CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# Load necessary components for RAG pipeline
embedding_function = get_embedding_function()

# Initialize Chroma vector store
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

# ChatGPT-like prompt template
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

# OpenAI model setup
model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")  # You can adjust the model and temperature if necessary


def prettify_response(raw_response, sources: list) -> str:
    """Prettify the raw response with source information."""

    # Extract the content from the raw_response, assuming it's an AIMessage object
    response_content = raw_response.content  # Direct access to the response content

    # Format the sources
    formatted_sources = "\n".join(
        ["{index}. {source}".format(index=i + 1, source=src.replace('\\', '/')) for i, src in enumerate(sources)]
    )

    # Create the prettified response
    prettified_response = """
{response_content}

Sources:
{formatted_sources}

""".format(response_content=response_content,
           formatted_sources=formatted_sources)

    return prettified_response


@cl.on_message
async def main(message: cl.Message):
    # Extract the text from the Chainlit message object
    query_text = message.content  # Ensure you're passing the actual message content (text) to the RAG pipeline

    # Retrieve documents from the vector store
    results = db.similarity_search_with_score(query_text, k=5)

    # Create context from retrieved documents
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # Format the prompt using context and the original query
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Generate the response using the OpenAI model
    raw_response = model.invoke(prompt)  # This returns an AIMessage object

    # Get document metadata (e.g., source IDs)
    sources = [doc.metadata.get("id", None) for doc, _score in results]

    # Prettify the raw response
    prettified_response = prettify_response(raw_response, sources)

    # Send the response back to the Chainlit interface
    await cl.Message(content=prettified_response).send()
