from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

load_dotenv()

#model1
llm1= HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature=0,
)
model = ChatHuggingFace(llm=llm1)

prompt1 = PromptTemplate(
    template = 'write a joke about the {topic}',
    input_variables={'topic'}
)

prompt2 = PromptTemplate(
    template = 'Exaplain the following jokes - {text}',
    input_variables={'text'}
)
parser = StrOutputParser()
chain = RunnableSequence(prompt1,model,parser,prompt2,model,parser)
chain.get_graph().print_ascii()
result_outputs = chain.invoke({'topic':'ai'})
print(result_outputs)

