##ChatBot for website
#1.get the LLM 
#2.get the Data from the company website
#3.make the data in chuncks
#4.create embedding for the data

#5.get the iput from user
#6.do a similarity search based on it
#7.give output X
#8.continue the conversation until the user types 0 X

#import libaraies
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import SeleniumURLLoader
from bs4 import BeautifulSoup as Soup
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from operator import itemgetter

#responses
def getoutput(x):
    response=(retriever | model).invoke(x)
    return response.context



MODEL='llama3:latest'

model=Ollama(model=MODEL)
embeddings = OllamaEmbeddings(model="nomic-embed-text",show_progress=True)

urls = [
    'https://grgmea.com',
    'https://grgmea.com/talent-acquisition-10-step-work-flow/'

]

loader = WebBaseLoader(urls)
docs = loader.load_and_split()

vector=FAISS.from_documents(docs,embeddings)
retriever= vector.as_retriever()

def user_interaction():
    question=''
    print(getoutput("hello what is grgmea?"))

user_interaction()

#retriever.invoke(input)


##from 5 to 8 Conversation with the chatbot

