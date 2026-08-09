from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature=0,
)

model = ChatHuggingFace(llm=llm)

#prompt template1 
template1 = PromptTemplate( 
    template = 'write a detailed report on the {topic}',
    input_variables=['topic']
)

#prompt template2 
template2 = PromptTemplate(
    template = 'write a five line summary on the following text. \n {text}',
    input_variables = ['text']
)


prompt1 = template1.invoke({'topic':'blackhole'})

result1 = model.invoke(prompt1)

prompt2 = template2.invoke({'text': result1.content})

result2 = model.invoke(prompt2)

print(result2.content)