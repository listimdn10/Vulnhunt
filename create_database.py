from llama_index.core import SimpleDirectoryReader
from llama_index.core .node_parser import SentenceSplitter 
from llama_index.core import Settings
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



if __name__ == "__main__":
    documents= load_documents()
    chunks = split_doc_to_chunk(documents)
 
