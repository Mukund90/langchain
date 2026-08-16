from langchain_core.tools import tool

#step1 create a functions 

# def multiply_two_numbers(x,y):
#     """multiply two numbers"""
#     return x * y 

# print(multiply_two_numbers(1,2))


# #step2 add the type hint 
# def multiply(a:int, b:int)->int:
#     """multiply two numbers"""
#     return a * b 

#This functions is become the Runnables 
#you pass the dictionary as inputs to the multiply Tools 
@tool
def multiply(a:int, b:int)->int:
    """multiply two numbers"""
    return a * b 


result = multiply.invoke({'a': 2 , 'b': 3})

print(result)

print(multiply.name)
print(multiply.description)
print(multiply.args)
print(multiply.args_schema.model_json_schema())