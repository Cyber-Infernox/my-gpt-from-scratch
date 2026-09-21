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

# --------------------------------------------------
# 15. Calculate loss
# --------------------------------------------------

loss = torch.nn.functional.cross_entropy(
    model_logits,
    y
)

print("loss:", loss)

# --------------------------------------------------
# 16. Calculate gradients
# --------------------------------------------------

linear_layer.zero_grad()

loss.backward()

# --------------------------------------------------
# 17. Update model parameters
# --------------------------------------------------

learning_rate = 0.01

with torch.no_grad():
    for parameter in linear_layer.parameters():
        parameter -= learning_rate * parameter.grad

# --------------------------------------------------
# 18. Train the model
# --------------------------------------------------

learning_rate = 0.01

for step in range(100):

    # 1. Make predictions
    model_logits = linear_layer(x_embeddings_with_position)

    # 2. Calculate loss
    loss = torch.nn.functional.cross_entropy(model_logits, y)

    # 3. Clear old gradients
    linear_layer.zero_grad()

    # 4. Calculate new gradients
    loss.backward()

    # 5. Update weights and bias
    with torch.no_grad():
        for parameter in linear_layer.parameters():
            parameter -= learning_rate * parameter.grad

    # 6. Print loss
    if step % 10 == 0:
        print("step:", step, "loss:", loss.item())

# --------------------------------------------------
# 19. Check the final loss
# --------------------------------------------------

model_logits = linear_layer(x_embeddings_with_position)

final_loss = torch.nn.functional.cross_entropy(
    model_logits,
    y
)

print("final loss:", final_loss.item())

# --------------------------------------------------
# 20. Get the model's prediction
# --------------------------------------------------

last_position_logits = model_logits[-1]

predicted_id = torch.argmax(last_position_logits)

predicted_character = id_to_char[predicted_id.item()]

print("predicted_id:", predicted_id)
print("predicted_character:", predicted_character)

# --------------------------------------------------
# 21. Convert last position logits to probabilities
# --------------------------------------------------

last_position_probs = torch.softmax(
    last_position_logits,
    dim=0
)

print("last_position_probs:", last_position_probs)
print("sum:", last_position_probs.sum())

# --------------------------------------------------
# 22. Sample from the model's probabilities
# --------------------------------------------------

sampled_id = torch.multinomial(
    last_position_probs,
    num_samples=1
)

sampled_character = id_to_char[sampled_id.item()]

print("sampled_id:", sampled_id)
print("sampled_character:", sampled_character)
