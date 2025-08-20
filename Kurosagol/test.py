from transformers import pipeline

model_name_or_path = 'Kurosawama/gemma-3-1b-it-Full-align'
pipe = pipeline("text-generation", model=model_name_or_path)
print(pipe("Translate the following set of premises to First-Order Logic: All men are mortal. Plato is a man.")[0]["generated_text"])