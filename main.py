from rag_pipeline import RAGPipeline
from evaluation import evaluate_retrieval

def main():
    rag = RAGPipeline()

    # ---------------------------
    # LOAD DATA
    # ---------------------------
    print("=" * 60)
    print("LOADING POPQA DATASET...")
    rag.load_data()

    print(f"Total Questions Loaded: {len(rag.questions)}")
    print(f"Total Corpus Passages: {len(rag.corpus)}")

    print("\nSample Question:")
    print(rag.questions[0])

    print("\nSample Answer:")
    print(rag.answers[0])

    # ---------------------------
    # BUILD INDEX
    # ---------------------------
    print("\n" + "=" * 60)
    print("BUILDING VECTOR + BM25 INDEX...")
    rag.build_index()

    print("Index built successfully.")
    print(f"Embedding Shape: {rag.embeddings.shape}")

    # ---------------------------
    # BASELINE RETRIEVAL EXAMPLES
    # ---------------------------
    print("\n" + "=" * 60)
    print("BASELINE DENSE RETRIEVAL EXAMPLES:")

    for i in range(3):
        query = rag.questions[i]
        indices = rag.dense_search(query, k=3)

        print(f"\nQuery {i+1}: {query}")
        for rank, idx in enumerate(indices):
            print(f"Top {rank+1}: {rag.corpus[idx]}")

    # ---------------------------
    # QUERY EXPANSION EXAMPLES
    # ---------------------------
    print("\n" + "=" * 60)
    print("QUERY EXPANSION EXAMPLES:")

    for i in range(3):
        original = rag.questions[i]
        expanded = rag.expand_query(original)

        print(f"\nOriginal: {original}")
        print(f"Expanded: {expanded}")

    # ---------------------------
    # HYBRID + RERANK EXAMPLES
    # ---------------------------
    print("\n" + "=" * 60)
    print("HYBRID SEARCH + RERANK EXAMPLES:")

    for i in range(3):
        query = rag.questions[i]

        hybrid_indices = rag.hybrid_search(query, k=5)
        reranked_indices = rag.rerank(query, hybrid_indices)

        print(f"\nQuery: {query}")

        print("Before Rerank:")
        for idx in hybrid_indices[:3]:
            print("-", rag.corpus[idx])

        print("After Rerank:")
        for idx in reranked_indices[:3]:
            print("-", rag.corpus[idx])

    # ---------------------------
    # EVALUATION
    # ---------------------------
    print("\n" + "=" * 60)
    print("RETRIEVAL METRICS:")
    results = evaluate_retrieval(rag)

    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    # ---------------------------
    # FINAL ANSWER GENERATION
    # ---------------------------
    print("\n" + "=" * 60)
    print("FINAL CITATION-GROUNDED ANSWERS:")

    for i in range(10):
        question = rag.questions[i]

        print(f"\nQuestion {i+1}: {question}")

        final_answer = rag.answer_question(question)

        print("Answer:")
        print(final_answer)

        print("-" * 40)


if __name__ == "__main__":
    main()