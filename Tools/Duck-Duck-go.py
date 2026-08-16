from langchain_community.tools  import DuckDuckGoSearchRun
from langchain_core.prompts import PromptTemplate

search_tools = DuckDuckGoSearchRun()

result = search_tools.invoke('what is the today news tell me')

print(result)