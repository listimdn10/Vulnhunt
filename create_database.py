from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

PDF_FOLDER="./dataset"

# Hàm tải tài liệu từ thư mục
def load_documents():
    loader = DirectoryLoader(PDF_FOLDER, glob="*.md")
    documents = loader.load()
    return documents

# Hàm chia nhỏ văn bản thành chunks
def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,  # Kích thước tối đa của mỗi chunk (số ký tự)
        chunk_overlap=500,  # Độ chồng lấp giữa các chunks
        length_function=len,
        add_start_index=True,  # Thêm thông tin về vị trí bắt đầu của chunk
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # In ví dụ về một chunk
    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks

if __name__ == "__main__":
    documents= load_documents()
    split_text(documents)

