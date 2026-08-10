from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel , RunnableBranch , RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Literal

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature=0,
)

model = ChatHuggingFace(llm =llm)

class FeedbackOutputParser(BaseModel):
    sentiment: Literal['positive','negative'] = Field(description='give the sentiment of the feedbacks')

parser2 = PydanticOutputParser(pydantic_object=FeedbackOutputParser)

prompt1 = PromptTemplate(
    template= 'classfiy the sentiment for the following feedbacks into text positive or Negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables = {'format_instruction': parser2.get_format_instructions()}
)

parser = StrOutputParser()

classifier_chains = prompt1| model | parser2 

prompt2 = PromptTemplate(
     template = 'Write an appropriate Response to this positive feedback \n {feedback}',
     input_variables=['feedback']
)

prompt3= PromptTemplate(
     template = 'Write an appropriate Response to this negative feedback \n {feedback}',
     input_variables=['feedback']
)

branch_chain = RunnableBranch( 
   (lambda x: x.sentiment == 'positive' , prompt2 | model | parser),
   (lambda x: x.sentiment == 'negative', prompt3 | model | parser ),
   RunnableLambda(lambda x: "could not find the sentiments")
)

chain = classifier_chains | branch_chain 

chain.get_graph().print_ascii()

result = chain.invoke({'feedback':'This is the terrible phone'})

print(result)