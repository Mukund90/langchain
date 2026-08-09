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

prompt = PromptTemplate(
    template = 'Generate five interesting five lines on the these {topics} ',
    input_variables=['topics']
)

parser = StrOutputParser()

chain = prompt | model | parser
result_outputs = chain.invoke({'topics': 'lions'})
# print(result_outputs)
chain.get_graph().print_ascii()
