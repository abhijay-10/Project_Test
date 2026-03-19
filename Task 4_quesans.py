from transformers import pipeline

qa_pipeline = pipeline("question-answering")

context = """
Artificial Intelligence is the simulation of human intelligence in machines.
It allows systems to learn, reason, and make decisions.
AI is widely used in healthcare, finance, and automation.
"""

question = "Where is AI widely used?"

result = qa_pipeline(question=question, context=context)

print("Answer:", result['answer'])


# Why transformers are better than RNNs?

# 1. Transformers process data in parallel, making them much faster than sequential RNNs.

# 2. They use attention mechanisms to capture long-range dependencies better.

# 3. This results in higher accuracy and scalability for NLP tasks.