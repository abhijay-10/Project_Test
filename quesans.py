from transformers import pipeline

# Load QA model explicitly (good practice)
qa_pipeline = pipeline(
    "text-question-answering",
    model="distilbert-base-cased-distilled-squad"
)

# Context
context = """
Artificial Intelligence is the simulation of human intelligence in machines.
It allows systems to learn, reason, and make decisions.
AI is widely used in healthcare, finance, and automation.
"""

# Question
question = "Where is AI widely used?"

# Inference
result = qa_pipeline(question=question, context=context)

# Output
print("Question:", question)
print("Answer:", result['answer'])
print("Confidence:", round(result['score'], 3))


# Why transformers are better than RNNs?

# 1. Transformers process data in parallel, making them much faster than sequential RNNs.

# 2. They use attention mechanisms to capture long-range dependencies better.

# 3. This results in higher accuracy and scalability for NLP tasks.