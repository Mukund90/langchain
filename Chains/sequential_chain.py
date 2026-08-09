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


prompt1 = PromptTemplate(
    template = 'Genearate a detailed report on the {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate( 
    template = 'Generate a 5 pointer summary form the following text \n {text}',
    input_variables=['text']
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser 
result = chain.invoke({'topic': 'Information Technology'})
print(result)
chain.get_graph().print_ascii()
