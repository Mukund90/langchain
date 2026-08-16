from  langchain_core.tools import tool


#Multiply functions
@tool
def multiply_two_numbers( a: int,b:int)->int: 
    """Multiply between two numbers"""
    return a * b 

#additions functions
@tool 
def additions_of_two_numbers(a:int, b:int)->int:
    """Additions of the two numbers"""
    return a + b

class MathToolkits:
    def get_Tools(self):
        return [multiply_two_numbers, additions_of_two_numbers]

toolkits = MathToolkits()
tools = toolkits.get_Tools()

for tool in tools: 
    print(tool.name,"=>" ,tool.description)