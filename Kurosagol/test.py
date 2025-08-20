from transformers import pipeline

model_name_or_path = 'Kurosawama/gemma-3-1b-it-Full-align'
pipe = pipeline("text-generation", model=model_name_or_path)
print(pipe("This movie was really")[0]["generated_text"])