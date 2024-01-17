import logging
import time

import retrieval


def main():
    qa = retrieval.get_qa()

    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break

        t0 = time.perf_counter()

        # Get the answer from the chain
        res = qa(query)
        answer, docs = res["result"], res["source_documents"]

        elapsed = time.perf_counter() - t0

        # Print the result
        print("\n\n> Question:")
        print(query)
        print(f"\n> Answer (elapsed: {int(elapsed)} seconds):")
        print(answer)

        # # Print the relevant sources used for the answer
        print(
            "----------------------------------SOURCE DOCUMENTS---------------------------"
        )
        for document in docs:
            print(
                f"\n> {document.metadata['source']} (page {document.metadata['page']}"
            )
            print(document.page_content)
        print(
            "----------------------------------SOURCE DOCUMENTS---------------------------"
        )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s",
        level=logging.INFO,
    )
    main()
