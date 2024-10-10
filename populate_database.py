import argparse
import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma  # Updated import from the new package
import re

CHROMA_PATH = "chroma"
DATA_PATH = "data"

def main():

    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()

    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents_flexibly(documents)
    add_to_chroma(chunks)

def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

def split_documents_flexibly(documents: list[Document]):
    section_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False  # No strict regex for sections
    )

    final_chunks = []

    # Splitting by paragraphs or sentences
    for doc in documents:
        text = doc.page_content  # Access the document content
        # First, attempt splitting by paragraph (if possible)
        paragraphs = re.split(r"\n\s*\n", text)  # Split by blank lines (paragraphs)

        if len(paragraphs) <= 1:
            # If the text does not contain paragraphs, fall back to sentence splitting
            sentences = re.split(r'(?<=[.!?]) +', text)  # Split by sentence boundaries
            chunks = section_splitter.split_text(" ".join(sentences))
        else:
            # Further split paragraphs if they're too long
            chunks = section_splitter.split_text(" ".join(paragraphs))

        # Create new Document objects from the split chunks
        for chunk in chunks:
            final_chunks.append(Document(page_content=chunk, metadata=doc.metadata))

    return final_chunks

def add_to_chroma(chunks: list[Document]):
    # Use the new `Chroma` from langchain_chroma package
    embeddings = get_embedding_function()

    # Initialize the Chroma vector store
    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,  # Automatically persists data
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = vector_store.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        vector_store.add_documents(new_chunks, ids=new_chunk_ids)
        # No need to call vector_store.persist(), since it's handled by the persist_directory
    else:
        print("✅ No new documents to add")

def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

if __name__ == "__main__":
    main()
