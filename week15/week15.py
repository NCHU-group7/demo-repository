from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import torch

# Define the context and question
context = """
The telephone was invented by Alexander Graham Bell in 1876. Bell was a Scottish-born inventor, scientist, 
and engineer best known for his work in the field of communication. He is credited with inventing the first 
practical telephone, which revolutionized global communication.
"""
question = "Who invented the telephone?"

# 1. BERT Answer (Fine-Tuned Pipeline)
qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")
bert_result = qa_pipeline(question=question, context=context)


# 2. Transformer Answer (Manual Processing)
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

# Tokenize inputs
inputs = tokenizer(question, context, return_tensors="pt", truncation=True)

# Forward pass through the model
outputs = model(**inputs)

# Get start and end logits (scores)
start_logits = outputs.start_logits
end_logits = outputs.end_logits

# Find the token positions with the highest scores
start_idx = torch.argmax(start_logits)
end_idx = torch.argmax(end_logits) + 1

# Decode the answer
answer_tokens = inputs["input_ids"][0][start_idx:end_idx]
answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

print(f"BERT Answer: {bert_result['answer']}")
print(f"Transformer Answer: {answer}")
