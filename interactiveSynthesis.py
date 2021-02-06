from transformers import pipeline
# import sys

pipe = pipeline('text-generation', model="dbmdz/german-gpt2",
                tokenizer="dbmdz/german-gpt2")

print("Bitte gib einen Satzanfang ein:")
input = input()

text = pipe(input, max_length=50)[0]["generated_text"]

print(text)
