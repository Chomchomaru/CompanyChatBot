##ChatBot for website
#1.get the LLM 
#2.get the Data from the company website
#3.make the data in chuncks
#4.create embedding for the data

#5.get the iput from user
#6.do a similarity search based on it
#7.give output X
#8.continue the conversation until the user types 0 X

APIKEY='llx-Zv1Ene56xuMc6S16xjUsom0YsGxvMNDrmJUjmFrpIF4tfvI2'

#import libaraies
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

parser=LlamaParse(
    api_key=APIKEY,
    result_type='markdown'
)


#information on the model, company and email
MODEL='llama3:latest'
model=Ollama(model=MODEL)
embeddings = OllamaEmbeddings(model="nomic-embed-text",show_progress=True)
company_name="GRG"
mail="zamin.khan2611@gmail.com"

from langchain.docstore.document import Document
#get the Data the data from the 4 pdf proved
#Llamaparse to parse the pdf
def read_data():
    reader = SimpleDirectoryReader(input_dir="./data")
    documents=reader.load_data()

    #change the format of the documnet to make it work
    docs=[Document(page_content=d.text )for d in documents]
    
    return docs

#take the document and the embedding model and embed and safe it in a db(for noew locally)
#create a vector db 
def embed():
    from langchain_community.vectorstores import FAISS
    doc=read_data()
    vec=FAISS.from_documents(doc,embeddings)
    return vec

#get the most similar vectors from the vector db 
#get the similar results
def similarity_search(vec,input):
    retervier=vec.as_retriever()
    ss=retervier.invoke(input)
    return ss

#use langchain to prompt engineer for the desired out
#get the template from this function
def create_template(input,similarity_results):
    from langchain.prompts import PromptTemplate

    Template=""" You are a virtual assistant in {company_name} company and are responsible for answer there question. Do not deviate to other topics which 
    are no related to the topic keep the answer short and clear and if there is no information regarding there enquire direct them to this mail:{mail}.
    here are some context and the main question guven by the user
    
    
    Context:{context}
    Question:{question}
    """

    prompt=PromptTemplate.from_template(Template)
    return prompt.format(company_name=company_name,mail=mail,context=similarity_results,question=input)

#ask the LLM about the query
def ask_llm(context):
    result=model.invoke(context)
    return result 

#the entry point
def chat(inputs):
    vecdb=embed()
    similarity_results=similarity_search(vecdb,inputs)
    ct=create_template(inputs,similarity_results)
    print(ask_llm(ct))

while True:
    i=input("Ask Mish about {company_name}?  ")
    if(i!='0'):
        chat(i)
    else: break