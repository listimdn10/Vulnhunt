from llama_index.core import SimpleDirectoryReader, ServiceContext, GPTVectorStoreIndex
from llama_index.core .node_parser import SentenceSplitter 
from llama_index.core import Settings
import openai 
from pinecone import Pinecone, ServerlessSpec
import os

from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import GPTVectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.openai import OpenAIEmbedding



# Hàm tải tài liệu từ thư mục
def load_documents():
    documents = SimpleDirectoryReader("dataset").load_data()
    print(len(documents))
    return documents  


# Hàm chia nhỏ văn bản thành chunks,  this class tries to keep sentences and paragraphs together
def split_doc_to_chunk(documents):
    # Khởi tạo TextSplitter
    text_splitter = SentenceSplitter(
        chunk_size=300,  # Kích thước tối đa mỗi chunk
        chunk_overlap=100,  # Độ chồng lấp giữa các chunk
    )
    # Split văn bản thành các chunk
    chunks = []
    for doc in documents:
        # print(f"Document length: {len(doc.text)}")
        # Split từng document thành các chunk
        split_chunks = text_splitter.split_text(doc.text)
        chunks.extend(split_chunks)
    
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    
    # In ví dụ về một chunk
    # print(chunks[2]) # In ra chunk đầu tiên
    return chunks


#tao index tren Pinecone
def upload_to_pinecone(): 
    # find API key in console at app.pinecone.io
    pc = Pinecone(api_key="pcsk_3g4mTY_HJ4TSsa17ZFrUDXxssVSKLxnVRZGvMiqUp84PrkrqeWF7YXhvbuADN5B4vnu1fz")
    # create the index if it does not exist already
    index_name = "vulnhunt-gpt"

    pc.create_index(
        name=index_name,
        dimension=1536, # Replace with your model dimensions
        metric="cosine", # Replace with your model metric
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ) 
    )
        # connect to the index
    pinecone_index = pc.Index(index_name)
    
    #create 
    # we can select a namespace (acts as a partition in an index)
    namespace = '' # default namespace
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    # setup our storage (vector db)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    import os
    # https://platform.openai.com/account/api-keys
    os.environ["OPENAI_API_KEY"] = "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"


    # setup the index/query process, ie the embedding model (and completion if used)
    embed_model = OpenAIEmbedding(model='text-embedding-ada-002', embed_batch_size=100)
    service_context = ServiceContext.from_defaults(embed_model=embed_model)

    index = GPTVectorStoreIndex.from_documents(
        documents, storage_context=storage_context,
        service_context=service_context
    )

if __name__ == "__main__":
    documents= load_documents()
    chunks = split_doc_to_chunk(documents)
    upload_to_pinecone()
