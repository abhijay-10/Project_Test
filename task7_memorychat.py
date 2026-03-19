from langchain_community.chat_models import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatOllama(model="mistral:latest", temperature=0.9)

chat_history = [
    SystemMessage(content='You are a helpful AI assistant')
]



while True:
    user_input = input('You: ')
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI: ",result.content)

print(chat_history)


"""Why is memory important in chat systems? (2–3 lines)

Memory is important in chat systems because it allows the model to retain context from previous interactions, making conversations more coherent and meaningful.
It helps in providing personalized and relevant responses instead of treating each query independently.
Without memory, the system behaves like a stateless bot, leading to disconnected and repetitive replies."""
