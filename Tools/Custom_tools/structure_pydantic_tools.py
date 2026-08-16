from langchain_core.tools import tool
from pydantic import BaseModel ,Field

class Multiply_Inputs(BaseModel):
    a : int = Field(required=True, description='The first Number')
    b: int = Field(required=True, description='The second Number')
