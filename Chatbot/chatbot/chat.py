from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv


load_dotenv()  
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# Extracts text from all pages of provided PDF documents
def get_pdf_text(pdf_docs):
    text = ""
    with open(pdf_docs, 'rb') as pdf:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
    return text

# Splits text into chunks of 10,000 characters with 1,000 character overlap
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=700)
    chunks = text_splitter.split_text(text)
    return chunks

# Creates and saves a FAISS vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


# Creates and returns a conversational chain for question answering
def get_conversational_chain():
    prompt_template = """
Answer the question concisely, focusing on the most relevant and important details from the PDF context. 
Refrain from mentioning any mathematical equations, even if they are present in provided context. 
Focus on the textual information available. Please provide direct quotations or references from PDF
to back up your response. If the answer is not found within the PDF, 
please state "answer is not available in the context."

If this is a follow-up question, start your response with "Continuing from the previous answer: ".

Context:\n {context}?\n
Question: \n{question}\n
Example response format:
Overview: 
(brief summary or introduction)
Key points: 
(point 1: paragraph for key details)
(point 2: paragraph for key details)
...
Use a mix of paragraphs and points to effectively convey the information.
"""
# Adjust temperature parameter to lower value to: 
# reduce model creativity & focus on factual accuracy
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=1.0)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)
    
    chain = get_conversational_chain()
    global response 

    response = chain.invoke(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    return("Reply: ", response["output_text"],"")

