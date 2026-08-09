from dotenv import load_dotenv

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import Optional
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature=0,
)

model = ChatHuggingFace(llm=llm)

#schema 
class Student(BaseModel):
    name : str = 'mukund'
    age : Optional[int] = None

new_students = {}
student = Student(**new_students)
print(student)