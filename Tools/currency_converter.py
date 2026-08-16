from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, ToolMessage 
from dotenv import load_dotenv
import requests
import os
load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

@tool
def currency_conversion(baseCurrency:str,targetCurrency:str)-> float:
    """ This functions fetches the currency conversion factor between against the baseCurrency and targetCurrency"""
    url= f"{os.getenv("CURRENCY_CONVERTER")}/{baseCurrency}/{targetCurrency}"
    response = requests.get(url)
    return response.json()

reponse_outputs = currency_conversion.invoke({"baseCurrency":"USD", "targetCurrency":"INR"})

print(reponse_outputs['conversion_rate'])


@tool
def currency_conversion_factor(baseCurrency:int, conversion_rate:float)->float: 
    """Given a currency conversion rate this functions calculate targetCurrency value from the baseCurrency """
    return baseCurrency * conversion_rate

outputs = currency_conversion_factor.invoke({"baseCurrency":10 ,"conversion_rate" : 95.5035})

print(outputs)


#tool binding
llm_binding_outputs = llm.bind_tools([currency_conversion,currency_conversion_factor])


#Human Messages 
query = "what is the conversion rate between the USD and INR , and based on that can you convert 10 USD to INR"
human_messages = HumanMessage(content=query)

#AIMessages 
AImessages = llm_binding_outputs.invoke(
    [human_messages]
)

tools_calls = AImessages.tool_calls[0]

currency_conversion = currency_conversion.invoke(
    tools_calls['args']
)
print(currency_conversion)
# conversion_rate_base_currency = currency_conversion['base_code']
conversion_rate_final = currency_conversion['conversion_rate']

currency_converter_final_data = currency_conversion_factor.invoke({
    "baseCurrency": 10,
    "conversion_rate": conversion_rate_final
})

print("\n final_result")
print(currency_converter_final_data)

tool_message = ToolMessage( 
    content = float(currency_converter_final_data),
    tool_call_id=tools_calls["id"]
)

final_result = llm.invoke(
    [
          human_messages,
          AImessages,
          tool_message
    ]
    
)

print(final_result.content)