import os
import torch
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class RagEngine:
    def __init__(self, persist_directory="./chroma_db", model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", embed_model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.persist_directory = persist_directory
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading Embedding Model: {embed_model_name} on {self.device}")
        self.embeddings = HuggingFaceEmbeddings(model_name=embed_model_name, model_kwargs={'device': self.device})
        
        # Initialize or load existing Chroma database
        print(f"Initializing Vector Database at: {persist_directory}")
        self.vectorstore = Chroma(persist_directory=persist_directory, embedding_function=self.embeddings)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

        print(f"Loading LLM: {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        if self.device == "cpu":
            self.model = self.model.to(self.device)

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=256,
            temperature=0.1,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            return_full_text=False
        )
        self.llm = HuggingFacePipeline(pipeline=self.pipe)
        
        # Setup Prompts
        self.prompt_template = """<|system|>
You are a helpful AI assistant. Use the following context to answer the question. If you don't know the answer, just say that you don't know, don't try to make up an answer.
Context:
{context}

<|user|>
{question}

<|assistant|>
"""
        self.prompt = PromptTemplate(template=self.prompt_template, input_variables=["context", "question"])
        
        # Create LCEL Chain
        self.llm_chain = self.prompt | self.llm | StrOutputParser()

    def add_knowledge(self, text: str):
        """Adds new knowledge text to the vector store."""
        documents = [Document(page_content=text, metadata={"source": "user_input"})]
        chunks = self.text_splitter.split_documents(documents)
        print(f"Adding {len(chunks)} chunks to Vector Database...")
        self.vectorstore.add_documents(chunks)
        print("Knowledge added successfully.")

    def query(self, query_text: str):
        """Queries the RAG system."""
        print(f"Querying: {query_text}")
        
        # Manually retrieving documents for the LCEL chain
        docs = self.vectorstore.similarity_search(query_text, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        
        # Invoke LCEL chain
        answer = self.llm_chain.invoke({"context": context, "question": query_text})
        
        sources = [doc.page_content for doc in docs]
        return {"answer": answer.strip(), "sources": sources}

if __name__ == "__main__":
    engine = RagEngine()
    print("Engine initialized.")
    
    # Test adding knowledge
    engine.add_knowledge("The secret password for the Inspire Global server is 'mango123'.")
    
    # Test querying
    res = engine.query("What is the secret password for the Inspire Global server?")
    print(f"Answer: {res['answer']}")
    print(f"Sources: {res['sources']}")
