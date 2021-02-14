from transformers import pipeline

text_generator = pipeline("./pegibert")

print(text_generator("As far as I am concerned, I will",
                     max_length=50, do_sample=False))
