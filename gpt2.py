import torch


# ==================================================
# 1. Training text
# ==================================================

text = "hello world"


# ==================================================
# 2. Create vocabulary
# ==================================================

characters = sorted(set(text))

char_to_id = {}

for i, character in enumerate(characters):
    char_to_id[character] = i

id_to_char = {}

for character, i in char_to_id.items():
    id_to_char[i] = character

vocab_size = len(characters)

print("characters:", characters)
print("char_to_id:", char_to_id)
print("vocab_size:", vocab_size)


# ==================================================
# 3. Encode text
# ==================================================

encoded_text = []

for character in text:
    encoded_text.append(char_to_id[character])

print("encoded_text:", encoded_text)


# ==================================================
# 4. Convert text to tensor
# ==================================================

data = torch.tensor(encoded_text)

print("data:", data)
print("data shape:", data.shape)


# ==================================================
# 5. Create all training examples
# ==================================================

block_size = 4

x = []
y = []

for i in range(len(data) - block_size):

    input_sequence = data[
        i : i + block_size
    ]

    target_sequence = data[
        i + 1 : i + block_size + 1
    ]

    x.append(input_sequence)
    y.append(target_sequence)


x = torch.stack(x)
y = torch.stack(y)

print("\nx:", x)
print("y:", y)

print("x shape:", x.shape)
print("y shape:", y.shape)


# ==================================================
# 6. Show training examples
# ==================================================

print("\nTraining examples:")

for i in range(len(x)):

    print(
        "input:",
        x[i],
        "target:",
        y[i]
    )


# ==================================================
# 7. Create token embedding table
# ==================================================

embedding_dim = 8

embedding_table = torch.randn(
    vocab_size,
    embedding_dim
)

print(
    "\nEmbedding table shape:",
    embedding_table.shape
)


# ==================================================
# 8. Create position embedding table
# ==================================================

position_embedding_table = torch.randn(
    block_size,
    embedding_dim
)

print(
    "Position embedding table shape:",
    position_embedding_table.shape
)


# ==================================================
# 9. Get token embeddings
# ==================================================

x_embeddings = embedding_table[x]

print(
    "x_embeddings shape:",
    x_embeddings.shape
)


# ==================================================
# 10. Get position embeddings
# ==================================================

positions = torch.arange(block_size)

position_embeddings = (
    position_embedding_table[positions]
)

print(
    "position_embeddings shape:",
    position_embeddings.shape
)


# ==================================================
# 11. Combine token + position embeddings
# ==================================================

x_embeddings_with_position = (
    x_embeddings + position_embeddings
)

print(
    "combined embeddings shape:",
    x_embeddings_with_position.shape
)