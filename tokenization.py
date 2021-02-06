from transformers import AutoTokenizer, AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained("dbmdz/german-gpt2")
model = AutoModelWithLMHead.from_pretrained("dbmdz/german-gpt2")

inputs = tokenizer("Die Wahlen der deutschen Medien sind entschieden: Trump muss verlieren. Bedauerlicherweise darf aber auch das amerikanische Volk mitreden, und das könnte es wagen, Sichtweisen zu vertreten, die hierzulande ignoriert werden.")

print(inputs)

print(tokenizer.tokenize("Hausaufgabenheft"))
