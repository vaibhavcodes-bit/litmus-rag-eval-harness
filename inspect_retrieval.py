from src.retrieval.retriever import retrieve_documents

question = "What is the recommended first step before starting a job search?"

docs = retrieve_documents(question, k=4)

print("RETRIEVED:", len(docs))

for i, doc in enumerate(docs, start=1):
    print()
    print(f"--- CHUNK {i} ---")
    print("SOURCE:", doc.metadata.get("source"))
    print("PAGE:", doc.metadata.get("page"))
    print("CONTENT:")
    print(doc.page_content)