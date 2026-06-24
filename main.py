from rag import ask_rag

query = "what is graph?"

result = ask_rag(query)

print(result["answer"])
print("Pages:", result["pages"])
print("Sources:", result["sources"])