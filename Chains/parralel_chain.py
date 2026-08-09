from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel

load_dotenv()

#model1
llm1= HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature=0,
)
model1 = ChatHuggingFace(llm=llm1)

#model2
llm2 = HuggingFaceEndpoint( 
    repo_id = "deepseek-ai/DeepSeek-V4-Flash-0731",
    task = 'text-generation',
    temperature = 0
)
model2 = ChatHuggingFace(llm=llm2)

#prompts1
prompts1 = PromptTemplate( 
    template = 'Generate a short and simple notes from the following text \n {text}',
    input_variables=['text']
)


#prompts2
prompt2 = PromptTemplate(
    template = 'Genearate a five quiz on the given text \n {text}',
    input_variables=['text']
) 

#prompts3
prompt3 = PromptTemplate( 
    template = 'Merge the provided notes and quiz into the single documents \n notes->{notes} and quiz->{quiz}',
    input_variables = ['notes', 'quiz']
)


parser = StrOutputParser()

#parallel_chains 
parallel_chains = RunnableParallel(
    {
        'notes' : prompts1| model1 | parser ,
        'quiz' :  prompt2 | model2 | parser , 
    }
)


text = """
Agent = Model + Harness. LangChain provides create_agent: a minimal, highly configurable harness. The harness is everything around the model loop: the prompt, the tools, and any middleware that shapes behavior. Start with the primitives and compose exactly what your use case needs. Supports OpenAI, Anthropic, Google, and more.
"""
#single_chains
final_result = chain = prompt3 | model2 | parser 
chain = parallel_chains | final_result

result = chain.invoke({'text' : text})
chain.get_graph().print_ascii()
print(result)
