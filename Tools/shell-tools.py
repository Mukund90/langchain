from langchain_community.tools import ShellTool

shellTools = ShellTool()

result = shellTools.invoke('ls -la')

print(result)