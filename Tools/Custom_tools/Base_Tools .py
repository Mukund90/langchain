from langchain_core.tools import BaseTool
from pydantic import BaseModel , Field
from typing import Type


class MultiplyInput(BaseModel):
    a : int = Field(required=True, description='The fist input is')
    b : int = Field(required=True, description='The second input is')


#This class is inherited from the base class 
class MultipltyTwoNUmbers(BaseTool):
    name: str = "multiply"
    description: str = "MUltiply Two Numbers"
    args_schema : Type[BaseModel] = MultiplyInput

    def _run(self,a :int, b:int)->int:
        return a*b 


Multiply_tools = MultipltyTwoNUmbers()

result = Multiply_tools.invoke({'a': 10 , 'b': 10})

print(result)
print(Multiply_tools.name)
print(Multiply_tools.description)