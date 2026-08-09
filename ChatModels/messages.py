from langchain_core.messages import SystemMessage , HumanMessage , AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature=0,
)
model = ChatHuggingFace(llm=llm)

messages = [SystemMessage(content= "you are the langchain devlopers"),
            HumanMessage(content = "Is the langchain is dominants in the 2026 ?"
           )]

result = model.invoke(messages)

messages.append(AIMessage(content = result.content))

print(messages)