import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text()
    return text

extracted_text = extract_text_from_pdf("About the company.pdf")


from langchain.text_splitter import RecursiveCharacterTextSplitter

text = extract_text_from_pdf("About the company.pdf")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_text(text)


from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
# 2. Initialize the HuggingFace Embedding Model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 3. Create FAISS Vector Store
vectorstore = FAISS.from_texts(chunks, embedding_model)

# 4. Save the vector database for reuse
vectorstore.save_local("faiss_index")


# filename: main.py

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Initialize FastAPI app
app = FastAPI()

# Load model and tokenizer once
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

# Request schema
class QuestionRequest(BaseModel):
    question: str
    

# Prompt template
def generateprompt(question):
    return f"""
You're a chatbot helping real estate customers. Answer politely and briefly. End with 'Let me know if I can help more!'.
context: {extracted_text}
Question: {question}
answer:
"""

# Core model logic
def ask(question):
    prompt = generateprompt(question)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# API route
@app.post("/ask")
def get_answer(req: QuestionRequest):
    answer = ask(req.question)
    return {"answer": answer}
