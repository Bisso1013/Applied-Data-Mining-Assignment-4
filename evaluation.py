def evaluate_retrieval(rag):
    correct = 0

    for q, a in zip(rag.questions, rag.answers):
        retrieved = rag.dense_search(q, k=1)

        if len(a) > 0 and rag.corpus[retrieved[0]] == a[0]:
            correct += 1

    recall_at_1 = correct / len(rag.questions)

    return {
        "Recall@1": recall_at_1
    }