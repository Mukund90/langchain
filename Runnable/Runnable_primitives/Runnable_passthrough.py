from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableSequence , RunnablePassthrough

load_dotenv()

llm1= HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature=0,
)

llm2 = HuggingFaceEndpoint( 
    repo_id = "deepseek-ai/DeepSeek-V4-Flash-0731",
    task = 'text-generation',
    temperature = 0
)

prompt1 = PromptTemplate( 
    template = 'Genearate a tweet about a {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template = 'Generate a Linkedin post  about a {topic}',
    input_variables={'topic'}
)

model1 = ChatHuggingFace(llm=llm1)
model2 = ChatHuggingFace(llm =llm2)

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt1 , model1, parser)
parralel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2, model2, parser)
})

final_chains = RunnableSequence(joke_gen_chain , parralel_chain)
result = final_chains.invoke({'topic':'Narendra Das Modi'})
print(result)