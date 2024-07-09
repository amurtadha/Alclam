from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

paths = [str(x) for x in Path("/workspace/ArNLP/datasets/plm/").glob("**/*.json")]

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()
print(paths)
# Customize training
tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

# Save files to disk
tokenizer.save_model("new_tokenizer/")
