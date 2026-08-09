from dotenv import load_dotenv

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature=0,
)

model = ChatHuggingFace(llm=llm)

chat_template = ChatPromptTemplate(
    [
        (
            "system",
            "You are a helpful {domain} expert."
        ),
        (
            "human",
            "Explain {topic} in a simple and easy-to-understand way."
        ),
    ]
)

prompt = chat_template.invoke(
    {
        "domain": "LangChain",
        "topic": "Models"
    }
)

response = model.invoke(prompt)

print(response.content)