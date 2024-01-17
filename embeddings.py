# Utility functions for Embeddings API with rate limiting
import time
from math import ceil
from typing import List

import vertexai
from langchain.embeddings import VertexAIEmbeddings
from pydantic import BaseModel

from constants import EMBED_NUM_BATCH, EMBED_QPM, LOCATION, PROJECT_ID

__INITIALIZED = False


def init_vertexai():
    global __INITIALIZED
    if not __INITIALIZED:
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        __INITIALIZED = True


init_vertexai()


def rate_limit(max_per_minute):
    period = 60 / max_per_minute
    while True:
        before = time.time()
        yield
        after = time.time()
        elapsed = after - before
        sleep_time = max(0, period - elapsed)

        sleep_time = ceil(sleep_time)

        if sleep_time > 0:
            print(".", end="")
            time.sleep(sleep_time)


class CustomVertexAIEmbeddings(VertexAIEmbeddings, BaseModel):
    num_instances_per_batch: int = EMBED_NUM_BATCH
    requests_per_minute: int = EMBED_QPM

    # Overriding embed_documents method
    def embed_documents(self, texts: List[str], batch_size: int = EMBED_NUM_BATCH):
        limiter = rate_limit(self.requests_per_minute)
        results = []
        docs = list(texts)

        while docs:
            # Working in batches because the API accepts maximum 5
            # documents per request to get embeddings
            head, docs = (
                docs[: self.num_instances_per_batch],
                docs[self.num_instances_per_batch :],
            )
            print(f"Getting embedding of {len(head)} docs; {len(docs)} left")
            chunk = self.client.get_embeddings(head)
            results.extend(chunk)
            next(limiter)

        return [r.values for r in results]
