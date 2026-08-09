from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature=0,
)

model = ChatHuggingFace(llm=llm)

chat_history = []

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    # Add the user's message
    chat_history.append(HumanMessage(content=user_input))

    # Send the complete conversation
    result = model.invoke(chat_history)

    # Print the AI response
    print("AI:", result.content)

    # Store the AI response
    chat_history.append(AIMessage(content=result.content))