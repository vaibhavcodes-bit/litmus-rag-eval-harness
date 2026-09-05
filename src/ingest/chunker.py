from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(documents):
    """
    Split loaded documents into smaller chunks.

    V1 configuration:
    - chunk_size: 500 characters
    - chunk_overlap: 50 characters
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    chunks = splitter.split_documents(documents)

    return chunks