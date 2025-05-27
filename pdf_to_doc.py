from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
import os
import nest_asyncio
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter


def pdf_to_doc(pdf_path):
    load_dotenv('.env')
    nest_asyncio.apply()
    # 파서 설정
    parser = LlamaParse(
        result_type="markdown",  # "markdown"과 "text" 사용 가능
        num_workers=8,  # worker 수 (default: 4)
        verbose=True,
        language="ko",
    )

    # SimpleDirectoryReader를 사용하여 파일 파싱
    file_extractor = {".pdf": parser}

    # LlamaParse로 파일 파싱
    documents = SimpleDirectoryReader(
        input_files=[pdf_path],
        file_extractor=file_extractor,
    ).load_data()

    import json

    # Document 객체를 딕셔너리로 변환하는 함수
    def document_to_dict(doc):
        return {
            'metadata': doc.metadata,
            'text': doc.text,
            'text_template': doc.text_template,
            'metadata_template': doc.metadata_template,
            'metadata_separator': doc.metadata_seperator
        }

    documents_dict = [document_to_dict(doc) for doc in documents]

    with open('documents.json', 'w', encoding='utf-8') as f:
        json.dump(documents_dict, f, ensure_ascii=False, indent=4)

    print("JSON 파일로 저장 완료!")

    doc_str = ''.join([doc.text for doc in documents])

    return doc_str


def load_docs(documents):
    # from langchain_text_splitters import KonlpyTextSplitter

    # text_splitter = KonlpyTextSplitter()
    # texts = text_splitter.split_text(documents)  # 한국어 문서를 문장 단위로 분할
    # print(texts[0])
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    # splits = text_splitter.split_documents(documents)
    splits = text_splitter.split_text(documents)
    return splits