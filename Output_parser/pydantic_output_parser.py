from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from pydantic import BaseModel ,Field
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature=0,
)

model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name: str = Field(description='Name of the persons')
    age : int = Field(gt=18 , description='Age of the persons')
    city : str = Field(description='Name of the city that belongs to')


parser = PydanticOutputParser(pydantic_object=Person)

template =PromptTemplate(
    template= 'generate the name age city of the fictional {name} person \n {format_instruction}',
    input_variables = ['name'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser
final_result = chain.invoke({'name': 'sri-lanka'})
print(final_result)
# prompt = template.invoke({'name': 'indian'})
# print("prompt",prompt)
# result = model.invoke(prompt)
# final_result = parser.parse(result.content)
# print(final_result)