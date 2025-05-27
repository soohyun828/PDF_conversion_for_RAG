import ollama
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma # chromadb
from langchain_ollama import OllamaEmbeddings # string embedding
import argparse
import os
from pdf_to_doc import pdf_to_doc, load_docs

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def parse_args():
    parser = argparse.ArgumentParser(description='Say hello')
    parser.add_argument('--pdf_path', default='./data/test.pdf', help='PDF Path to convert')
    parser.add_argument('--doc_path', default=None, help='Json Path converted')
    parser.add_argument('--db_directory', default='db/test_embedding_db', help='Json Path converted')
    

    return parser.parse_args()


# make vector DB
def make_vectordb(args, docs):
    embeddings = OllamaEmbeddings(model='my_llm:latest') # role: str to embedding vectors
    # vectorstore = Chroma("langchain_store", embeddings)
    # 벡터 스토어에 문서와 벡터 저장
    persist_directory = args.db_directory
    vectordb = Chroma.from_texts(docs, embedding=embeddings, persist_directory=persist_directory)
    vectordb.persist() # 데이터베이스 disk에 저장
    # 저장된 벡터 스토어 로드
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    print(f'All docs stored at {persist_directory}')

    return vectordb


def main():
    
    args = parse_args()
    query = '남성 관련 질환에 대한 주요 보장 내용 알려줘.'

    # load vectorDB
    if os.path.isdir(args.db_directory): # vector DB 저장되어 있으면
        embeddings = OllamaEmbeddings(model='my_llm:latest') # role: str to embedding vectors
        vectordb = Chroma(persist_directory=args.db_directory, embedding_function=embeddings)
    else: # save vectorDB from document
        # convert pdf to str
        if args.doc_path is not None:
            import json
            with open(args.doc_path, 'r', encoding='UTF-8') as f:
                documents = json.load(f)
            doc_str = ''.join([doc['text'] for doc in documents])
        else:
            doc_str = pdf_to_doc(args.pdf_path)
        # seperate string for each chunks
        chunks = load_docs(doc_str)
        vectordb = make_vectordb(chunks)        
    
    # retriever = vectordb.as_retriever(search_kwargs={'k': 1})
    # print(f'질문과 가장 유사한 내용은 {retriever.invoke(query)}입니다.')
    
    llm = ChatOllama(model='my_llm:latest',temperature=0)
    template = ''' context를 기반으로 질문에 한국어로 간결하게 대답해줘. Context : {context}, Question: {question} '''
    prompt = ChatPromptTemplate.from_template(template)
    retriever = vectordb.as_retriever()
    llm_rag = {'context': retriever, 'question': RunnablePassthrough()} | prompt | llm | StrOutputParser()
    print(f'temperature O) => {llm_rag.invoke(query)}')
    print(llm_rag.get_graph().print_ascii())


if __name__ == '__main__':
    main()