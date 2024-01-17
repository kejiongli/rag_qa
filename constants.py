import os
from pathlib import Path

from langchain.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredExcelLoader,
)

PROJECT_ID = os.environ["GCLOUD_PROJECT_ID"]
LOCATION = "europe-west2"

ROOT_DIRECTORY = Path(__file__).parent.absolute()

SOURCE_DIRECTORY = ROOT_DIRECTORY / "data"

# Define the folder for storing database
PERSIST_DIRECTORY = ROOT_DIRECTORY / "DB"

TOP_DOCUMENTS = 5
SCORE_THRESHOLD = 0.55
TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 128
TOP_P = 0.95
TOP_K = 40

EMBED_QPM = 60  # 60 requests per minute
EMBED_NUM_BATCH = 5  # max number of documents per request
CHUNK_SIZE = 3000
CHUNK_OVERLAP = 300

EMBEDDING_MODEL = "textembedding-gecko@001"
VERTEX_LLM_MODEL = "text-bison@001"

# Prompt template
PROMPT_TEMPLATE = """
Answer the question given the context below as {{Context:}}. \n
If the answer is not available in the {{Context:}} and you are not confident about the output,
please say "Information not available in provided context". \n\n
Context: {context}\n
Question: {question} \n
Answer:
"""

# not a very good example:
# PROMPT_TEMPLATE = """You are a helpful, respectful and honest assistant. \n
# You write in an eloquent, professional, and polite tone. \n
# You do not thank the user for their question, just focus on the answer to the question. \n
# Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. \n
# Use the following pieces of context as {{Context:}} to answer the customer's question at the end. \n
# If you don't know the answer, just say that you don't know, don't try to make up an answer. \n
# If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. \n
# Do not repeat sentences. \n
# Respond in full, grammatically-correct sentences.\n
#
# Context: {context}?
# Question: {question}
# Helpful answer.
# """

# https://python.langchain.com/en/latest/_modules/langchain/document_loaders/excel.html#UnstructuredExcelLoader
DOCUMENT_MAP = {
    ".txt": TextLoader,
    ".py": TextLoader,
    ".pdf": PyPDFLoader,
    ".csv": CSVLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlxs": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
}

# Can be changed to a specific number
INGEST_THREADS = os.cpu_count() or 8
