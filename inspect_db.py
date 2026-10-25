import chromadb

def inspect():
    # Path must match the one in rag_engine.py
    persist_directory = "./chroma_db"
    
    client = chromadb.PersistentClient(path=persist_directory)
    
    # LangChain's default collection name for Chroma is usually 'langchain'
    try:
        collection = client.get_collection("langchain")
        count = collection.count()
        print(f"--- ChromaDB Inspection ---")
        print(f"Total entries in 'langchain' collection: {count}\n")
        
        # Get the documents
        data = collection.get(limit=10) # Limit to 10 for readability
        
        for i in range(len(data['documents'])):
            print(f"ID: {data['ids'][i]}")
            print(f"Content: {data['documents'][i][:200]}...") # Truncate for display
            print(f"Metadata: {data['metadatas'][i]}")
            print("-" * 30)
            
    except Exception as e:
        print(f"Error accessing collection: {e}")
        print("Collections available:", client.list_collections())

if __name__ == "__main__":
    inspect()
