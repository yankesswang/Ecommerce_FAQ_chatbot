from langchain.prompts import ChatPromptTemplate
from langserve import RemoteRunnable

chtabot = RemoteRunnable("http://localhost:9000/chatbot/")
response = chtabot.invoke({"input": "Hi there"})
print(response)