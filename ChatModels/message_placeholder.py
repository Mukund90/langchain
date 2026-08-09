from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful customer support agent."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        (
            "human",
            "{query}"
        )
    ]
)

chat_history = []

with open("chat_history.txt", "r") as f:
    for line in f:
        role, message = line.strip().split(":", 1)

        if role == "human":
            chat_history.append(HumanMessage(content=message.strip()))
        elif role == "ai":
            chat_history.append(AIMessage(content=message.strip()))

prompt = chat_template.invoke(
    {
        "chat_history": chat_history,
        "query": "Where is my refund?"
    }
)

print(prompt)