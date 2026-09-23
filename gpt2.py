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

# ==================================================
# 12. Create linear layer
# ==================================================

linear_layer = torch.nn.Linear(
    embedding_dim,
    vocab_size
)

print("\nLinear layer:", linear_layer)


# ==================================================
# 13. Create model logits
# ==================================================

model_logits = linear_layer(
    x_embeddings_with_position
)

print(
    "model_logits shape:",
    model_logits.shape
)

# ==================================================
# 14. Calculate loss
# ==================================================

B, T, C = model_logits.shape

print("\nB:", B)
print("T:", T)
print("C:", C)

# ==================================================
# 15. Convert logits to probabilities
# ==================================================

model_probs = torch.softmax(
    model_logits,
    dim=2
)

print(
    "\nmodel_probs shape:",
    model_probs.shape
)

print(
    "probability sum:",
    model_probs[0, 0].sum()
)

# ==================================================
# 16. Calculate loss
# ==================================================

B, T, C = model_logits.shape

print("\nB:", B)
print("T:", T)
print("C:", C)


model_logits_flat = model_logits.view(
    B * T,
    C
)

y_flat = y.view(
    B * T
)

print(
    "model_logits_flat shape:",
    model_logits_flat.shape
)

print(
    "y_flat shape:",
    y_flat.shape
)


loss = torch.nn.functional.cross_entropy(
    model_logits_flat,
    y_flat
)

print("loss:", loss.item())

# ==================================================
# 17. Train the model
# ==================================================

learning_rate = 0.01

for step in range(1000):

    # Make predictions
    model_logits = linear_layer(
        x_embeddings_with_position
    )

    # Get the shape
    B, T, C = model_logits.shape

    # Flatten predictions and targets
    model_logits_flat = model_logits.view(
        B * T,
        C
    )

    y_flat = y.view(
        B * T
    )

    # Calculate loss
    loss = torch.nn.functional.cross_entropy(
        model_logits_flat,
        y_flat
    )

    # Remove old gradients
    linear_layer.zero_grad()

    # Calculate new gradients
    loss.backward()

    # Update parameters
    with torch.no_grad():

        for parameter in linear_layer.parameters():

            parameter -= learning_rate * parameter.grad

    # Print loss every 100 steps
    if step % 100 == 0:

        print(
            "step:",
            step,
            "loss:",
            loss.item()
        )

# ==================================================
# 18. Get model predictions
# ==================================================

predicted_ids = torch.argmax(
    model_logits,
    dim=2
)

print("\npredicted_ids:")
print(predicted_ids)

print("predicted_ids shape:", predicted_ids.shape)

# ==================================================
# 19. Convert predicted IDs to characters
# ==================================================

print("\nPredictions:")

for i in range(len(predicted_ids)):

    predicted_text = ""

    for token_id in predicted_ids[i]:

        predicted_text += id_to_char[token_id.item()]

    print(
        "predicted:",
        predicted_text
    )

# ==================================================
# 20. Compare predictions with targets
# ==================================================
correct = predicted_ids == y

print("\nCorrect predictions:")
print(correct)

# ==================================================
# 21. Show input, target, and prediction
# ==================================================
print("\nDetailed predictions:")

for i in range(len(x)):

    input_text = ""
    target_text = ""
    predicted_text = ""

    for token_id in x[i]:
        input_text += id_to_char[token_id.item()]

    for token_id in y[i]:
        target_text += id_to_char[token_id.item()]

    for token_id in predicted_ids[i]:
        predicted_text += id_to_char[token_id.item()]

    print(
        "input:",
        input_text,
        "| target:",
        target_text,
        "| predicted:",
        predicted_text
    )