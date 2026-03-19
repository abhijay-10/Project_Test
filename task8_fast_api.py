# pip install fastapi uvicorn langchain langchain-core langchain-community

from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama

app = FastAPI()


llm = ChatOllama(model="mistral:latest")  

class ChatRequest(BaseModel):
    message: str

class SummarizeRequest(BaseModel):
    text: str


chat_prompt = PromptTemplate(
    input_variables=["message"],
    template="Reply in a helpful and concise way:\n\nUser: {message}\nAI:"
)

summarize_prompt = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text in short:\n\n{text}"
)


@app.post("/chat")
def chat(req: ChatRequest):
    prompt = chat_prompt.format(message=req.message)
    response = llm.invoke(prompt)

    return {
        "user_message": req.message,
        "response": response.content
    }


@app.post("/summarize")
def summarize(req: SummarizeRequest):
    prompt = summarize_prompt.format(text=req.text)
    response = llm.invoke(prompt)

    return {
        "summary": response.content
    }

