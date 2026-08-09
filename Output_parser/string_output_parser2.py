from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
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

parser = StrOutputParser()
#chain 
chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic': 'blackhole'})

print(result)