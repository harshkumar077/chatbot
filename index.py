import os
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
from PyPDF2 import PdfReader

# Set OpenAI API Key directly
os.environ['OPENAI_API_KEY'] = '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'

def create_index(file_path: str) -> None:
    reader = PdfReader(file_path)
    text = ''
    for page in reader.pages:
        text += page.extract_text()

    with open('output.txt', 'w', encoding='utf-8') as file:
        file.write(text)

    loader = DirectoryLoader(
        './',
        glob='**/*.txt',
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'}
    )

    documents = loader.load()

    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1024,
        chunk_overlap=128
    )

    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv('OPENAI_API_KEY')
    )

    persist_directory = 'db'

    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    vectordb.persist()

create_index('sample.pdf')
