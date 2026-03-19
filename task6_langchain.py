
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama

template = """
Explain the following topic in a simple and clear way:
Topic: {topic}
Explanation:
"""

prompt = PromptTemplate(
    input_variables=["topic"],
    template=template
)

llm = ChatOllama(
    model="mistral:latest",   # ollama run mistral:latest
    temperature=0.7
)

user_topic = input("Enter a topic: ")

# Format prompt---format karne se prompt structured way mei aa jata hai
final_prompt = prompt.format(topic=user_topic)

response = llm.invoke(final_prompt)

print("\nGenerated Explanation:\n")
print(response.content)


# Role of PromptTemplate in LangChain:

# 1. PromptTemplate helps create dynamic and reusable prompts by inserting variables (like topic).

# 2. It ensures consistent structure for inputs to the LLM.

# 3. This improves response quality and maintainability in applications.