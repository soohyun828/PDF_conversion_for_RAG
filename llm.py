import ollama
import chromadb
import argparse
from pdf_to_doc import pdf_to_doc, load_docs

def parse_args():
    parser = argparse.ArgumentParser(description='Say hello')
    parser.add_argument('--pdf_path', default='./data/test.pdf', help='PDF Path to convert')
    parser.add_argument('--doc_path', default=None, help='Json Path converted')

    return parser.parse_args()





def main():
    
    args = parse_args()

    # convert pdf to str
    if args.doc_path is not None:
        import json
        with open(args.doc_path, 'r') as f:
            documents = json.load(f)
        doc_str = ''.join([doc['text'] for doc in documents])
    else:
        doc_str = pdf_to_doc(args.pdf_path)
    
    chunks = load_docs(doc_str)


if __name__ == '__main__':
    main()