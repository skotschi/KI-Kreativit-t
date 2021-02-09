from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase, NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

# Loading File
with open('pegida_korpus.txt') as f:
    pegidaCorpus = f.read()

# First we create an empty Byte-Pair Encoding model (i.e. not trained model)
tokenizer = Tokenizer(BPE())

tokenizer.normalizer = Sequence([
    NFKC(),
    Lowercase()
])

# Our tokenizer also needs a pre-tokenizer responsible for converting the input to a ByteLevel representation.
tokenizer.pre_tokenizer = ByteLevel()

# And finally, let's plug a decoder so we can recover from a tokenized input to the original one
tokenizer.decoder = ByteLevelDecoder()

# We initialize our trainer, giving him the details about the vocabulary we want to generate
trainer = BpeTrainer(vocab_size=25000, show_progress=True,
                     initial_alphabet=ByteLevel.alphabet())
tokenizer.train(trainer, ["pegida_korpus.txt"])

print("Trained vocab size: {}".format(tokenizer.get_vocab_size()))

tokenizer.model.save('./pegibert')

tokenizer.model = BPE('./pegibert/vocab.json', './pegibert/merges.txt')
encoding = tokenizer.encode(
    "Ich wiederhole, aber das ist die Wahrheit. Auch Vollverschleierung ist ein schleichender Prozess. Der Weg zur Dunkelheit passiert nicht von heute auf Morgen. Aber am Ende steht immer die Finsternis.")


print("Encoded string: {}".format(encoding.tokens))

decoded = tokenizer.decode(encoding.ids)
print("Decoded string: {}".format(decoded))

text_generator = pipe
