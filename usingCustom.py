from tokenizers import Tokenizer
from transformers import pipeline
from tokenizers.models import BPE

text_generator = pipeline("text-generation")

print(text_generator("As far as I am concerned, I will",
                     max_length=50, do_sample=False))

loaded_model = TFDistilBertForSequenceClassification.from_pretrained(
    "/tmp/sentiment_custom_model")
