import logging
import os
from concurrent.futures import (
    as_completed,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
)
from pathlib import Path
from typing import Dict, List, Tuple

from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

from constants import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENT_MAP,
    EMBEDDING_MODEL,
    INGEST_THREADS,
    PERSIST_DIRECTORY,
    SOURCE_DIRECTORY,
)
from embeddings import CustomVertexAIEmbeddings


def load_single_document(file_path: str) -> List[Document]:
    # Loads a single document from a file path
    loader_class = DOCUMENT_MAP.get(Path(file_path).suffix)
    if not loader_class:
        raise ValueError("Document type is undefined")

    loader = loader_class(file_path)

    if loader_class is PyPDFLoader:
        pages = loader.load_and_split()
    else:
        pages = loader.load()[0]
    return pages if isinstance(pages, list) else [pages]


def load_document_batch(filepaths: List[str]) -> Dict[str, List[Document]]:
    logging.info("Loading document batch")
    # create a thread pool
    with ThreadPoolExecutor(len(filepaths)) as exe:
        # load files
        futures = [exe.submit(load_single_document, f) for f in filepaths]
        # collect data
        data_list = [future.result() for future in futures]
        # return data and file paths
        return dict(zip(filepaths, data_list))


def load_documents(source_dir: Path) -> List[Document]:
    # Loads all documents from the source documents directory
    paths = [str(f) for f in source_dir.glob("**/*") if f.suffix in DOCUMENT_MAP]

    # Have at least one worker and at most INGEST_THREADS workers
    n_workers = min(INGEST_THREADS, max(len(paths), 1))
    chunksize = round(len(paths) / n_workers)
    docs = []
    with ProcessPoolExecutor(n_workers) as executor:
        futures: List[Future[Dict[Path, List[Document]]]] = []
        # split the load operations into chunks
        for i in range(0, len(paths), chunksize):
            # select a chunk of filenames
            filepaths = paths[i : (i + chunksize)]
            # submit the task
            future = executor.submit(load_document_batch, filepaths)
            futures.append(future)

        # process all results
        for future in as_completed(futures):
            docs.extend([x for fs in future.result().values() for x in fs])

    return docs


def split_documents(documents: List[Document]) -> Tuple[List[Document], List[Document]]:
    # Splits documents for correct Text Splitter
    text_docs, python_docs = [], []
    for doc in documents:
        file_extension = os.path.splitext(doc.metadata["source"])[1]
        if file_extension == ".py":
            python_docs.append(doc)
        else:
            text_docs.append(doc)

    return text_docs, python_docs


def main():
    # Load documents and split in chunks
    logging.info(f"Loading documents from {SOURCE_DIRECTORY}")
    documents = load_documents(SOURCE_DIRECTORY)
    text_documents, python_documents = split_documents(documents)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    texts = text_splitter.split_documents(text_documents)
    texts.extend(python_splitter.split_documents(python_documents))
    logging.info(f"Loaded {len(texts)} documents from {SOURCE_DIRECTORY}")
    logging.info(f"Split into {len(texts)} chunks of text")

    # Create embeddings
    vertex_embeddings = CustomVertexAIEmbeddings(model_name=EMBEDDING_MODEL)

    db = FAISS.from_documents(documents=texts, embedding=vertex_embeddings)

    # Store db to local
    db.save_local(PERSIST_DIRECTORY)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s",
        level=logging.INFO,
    )
    main()
