from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model=".qberto",
    tokenizer=".qberto"
)

# The sun <mask>.
# =>

result = fill_mask("La suno <mask>.")
