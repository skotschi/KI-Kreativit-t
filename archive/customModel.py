import json
from tokenizers import ByteLevelBPETokenizer

tweetTexts = []

with open('posts.json') as f:
    tweets = json.load(f)

# for tweet in tweets:
#     print(tweet['text'])

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(files="posts.json", vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

# Save files to disk
tokenizer.save_model(".", "qberto")
