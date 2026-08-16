from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, ToolMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

#original tools
@tool
def multiply_Two_Numbers(a: int, b: int):
    """Given Functions calculate multiply between two Numbers"""
    return a * b


#bind the tools
llm_with_tools = llm.bind_tools([multiply_Two_Numbers])


#query
query = "Can you multiply 10 with 12?"

#human Messages
human_message = HumanMessage(content=query)


#ai_Messages
ai_message = llm_with_tools.invoke([
    human_message
])
print("\n ai messages")
print(ai_message)
tool_call = ai_message.tool_calls[0]


#Tools calling 
result = multiply_Two_Numbers.invoke(
    tool_call["args"]
)

#passing to the llm with the ans
tool_message = ToolMessage(
    content=str(result),
    tool_call_id=tool_call["id"]
)

result_data = llm.invoke([
    human_message,
    ai_message,
    tool_message
])


print("\nFinal LLM response:")
print(result_data.content)