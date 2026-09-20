import torch

# --------------------------------------------------
# 1. Training text
# --------------------------------------------------

text = "hello world"


# --------------------------------------------------
# 2. Create vocabulary
# --------------------------------------------------

characters = sorted(set(text))

print("characters:", characters)


# --------------------------------------------------
# 3. Character → ID
# --------------------------------------------------

char_to_id = {}

for i, character in enumerate(characters):
    char_to_id[character] = i

print("char_to_id:", char_to_id)


# --------------------------------------------------
# 4. ID → Character
# --------------------------------------------------

id_to_char = {}

for character, i in char_to_id.items():
    id_to_char[i] = character

print("id_to_char:", id_to_char)


# --------------------------------------------------
# 5. Encode text
# --------------------------------------------------

encoded_text = []

for character in text:
    encoded_text.append(char_to_id[character])

print("encoded_text:", encoded_text)


# --------------------------------------------------
# 6. Convert encoded text to PyTorch tensor
# --------------------------------------------------

data = torch.tensor(encoded_text)

print("data:", data)
print("data dtype:", data.dtype)
print("data shape:", data.shape)


# --------------------------------------------------
# 7. Create input and target sequences
# --------------------------------------------------

block_size = 4

x = data[:block_size]
y = data[1:block_size + 1]

print("x:", x)
print("y:", y)


# --------------------------------------------------
# 8. Show individual training examples
# --------------------------------------------------

for t in range(block_size):
    context = x[:t + 1]
    target = y[t]

    print("context:", context, "target:", target)


# --------------------------------------------------
# 9. Vocabulary size
# --------------------------------------------------

vocab_size = len(characters)

print("vocab_size:", vocab_size)


# --------------------------------------------------
# 10. Create random logits
# --------------------------------------------------

logits = torch.randn(vocab_size)

print("logits:", logits)
print("logits shape:", logits.shape)


# --------------------------------------------------
# 11. Convert logits to probabilities
# --------------------------------------------------

probs = torch.softmax(logits, dim=0)

print("probabilities:", probs)
print("probabilities sum:", probs.sum())


# --------------------------------------------------
# 12. Sample a token from the probabilities
# --------------------------------------------------

next_id = torch.multinomial(probs, num_samples=1)

print("next_id:", next_id)


# --------------------------------------------------
# 13. Convert token ID back to character
# --------------------------------------------------

next_character = id_to_char[next_id.item()]

print("next_character:", next_character)

input_token = x[0]

print("input_token:", input_token)
print("input_token shape:", input_token.shape)

embedding_dim = 8

embedding_table = torch.randn(vocab_size, embedding_dim)

print("embedding_table:")
print(embedding_table)
print("embedding_table shape:", embedding_table.shape)

x_embeddings = embedding_table[x]

print("x:", x)
print("x_embeddings:")
print(x_embeddings)
print("x_embeddings shape:", x_embeddings.shape)

position_embedding_table = torch.randn(block_size, embedding_dim)

print("position_embedding_table:")
print(position_embedding_table)
print("position_embedding_table shape:", position_embedding_table.shape)

position_embeddings = position_embedding_table[
    torch.arange(block_size)
]

print("position_embeddings:")
print(position_embeddings)
print("position_embeddings shape:", position_embeddings.shape)

x_embeddings_with_position = x_embeddings + position_embeddings

print("x_embeddings_with_position:")
print(x_embeddings_with_position)
print("shape:", x_embeddings_with_position.shape)

linear_layer = torch.nn.Linear(embedding_dim, vocab_size)

print("linear_layer:")
print(linear_layer)

# --------------------------------------------------
# 10. Pass embeddings through the linear layer
# --------------------------------------------------

model_logits = linear_layer(x_embeddings_with_position)

print("model_logits:")
print(model_logits)
print("model_logits shape:", model_logits.shape)

# --------------------------------------------------
# 11. Convert model logits to probabilities
# --------------------------------------------------

model_probs = torch.softmax(model_logits, dim=1)

print("model_probs:")
print(model_probs)
print("model_probs shape:", model_probs.shape)

# --------------------------------------------------
# 12. Get probabilities for the last position
# --------------------------------------------------

last_position_probs = model_probs[-1]

print("last_position_probs:")
print(last_position_probs)
print("shape:", last_position_probs.shape)
print("sum:", last_position_probs.sum())

# --------------------------------------------------
# 13. Sample the next token
# --------------------------------------------------

next_id = torch.multinomial(last_position_probs, num_samples=1)

print("next_id:", next_id)

# --------------------------------------------------
# 14. Convert sampled token ID back to character
# --------------------------------------------------

next_character = id_to_char[next_id.item()]

print("next_character:", next_character)