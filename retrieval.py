import logging
from functools import lru_cache

from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import VertexAI
from langchain.vectorstores import FAISS

from constants import (
    EMBEDDING_MODEL,
    MAX_OUTPUT_TOKENS,
    PERSIST_DIRECTORY,
    PROMPT_TEMPLATE,
    SCORE_THRESHOLD,
    TEMPERATURE,
    TOP_DOCUMENTS,
    TOP_K,
    TOP_P,
    VERTEX_LLM_MODEL,
)
from embeddings import CustomVertexAIEmbeddings


@lru_cache(maxsize=16)
def get_qa(
    *,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
    top_k: int = TOP_K,
    prompt_template=PROMPT_TEMPLATE,
    top_num_retrieve_doc: int = TOP_DOCUMENTS,
    score_threshold: float = SCORE_THRESHOLD,
):
    """
    1. Loads an embedding model, can be HuggingFaceInstructEmbeddings or HuggingFaceEmbeddings
    2. Loads the existing vector-store that was created by inget.py
    3. Loads the local LLM using load_model function - You can now set different LLMs.
    4. Returns the Question Answer retrieval chain.
    """

    logging.info(f"Starting Query and Answering...")

    custom_embeddings = CustomVertexAIEmbeddings(model_name=EMBEDDING_MODEL)

    # Load local vector store
    vector_store = FAISS.load_local(PERSIST_DIRECTORY, custom_embeddings)

    # Init retriever. Asking for just 1 document back
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": top_num_retrieve_doc, "score_threshold": score_threshold},
    )

    llm = VertexAI(
        model_name=VERTEX_LLM_MODEL,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        verbose=True,
    )
    # llm = load_model(model_name=VERTEX_LLM_MODEL)

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain_type_kwargs = {"prompt": prompt}

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
    )

    print(
        f"Creating QA using {top_p=} {temperature=} {top_k=} {max_output_tokens=} {top_num_retrieve_doc} {score_threshold} id={id(qa)} llm={id(llm)}"
    )
    return qa
