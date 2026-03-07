import chromadb
import os

db_path = os.path.join(os.getcwd(), ".refinery", "vector_db")
client = chromadb.PersistentClient(path=db_path)

print(f"Checking Path: {db_path}")
collections = client.list_collections()
print(f"Found {len(collections)} collections: {[c.name for c in collections]}")

for coll_name in [c.name for c in collections]:
    coll = client.get_collection(name=coll_name)
    print(f"Collection: {coll_name} | Chunks: {coll.count()}")
    if coll.count() > 0:
        sample = coll.get(limit=1)
        print(f" - Sample Text: {sample['documents'][0][:50]}...")