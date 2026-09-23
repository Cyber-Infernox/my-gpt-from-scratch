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
# 5. Create input and target sequences
# ==================================================

block_size = 4

x = data[:block_size]
y = data[1:block_size + 1]

print("x:", x)
print("y:", y)


# ==================================================
# 6. Show training examples
# ==================================================

print("\nTraining examples:")

for t in range(block_size):
    context = x[:t + 1]
    target = y[t]

    print("context:", context, "target:", target)


# ==================================================
# 7. Create token embedding table
# ==================================================

embedding_dim = 8

embedding_table = torch.randn(
    vocab_size,
    embedding_dim
)

print("\nEmbedding table shape:", embedding_table.shape)


# ==================================================
# 8. Create position embedding table
# ==================================================

position_embedding_table = torch.randn(
    block_size,
    embedding_dim
)

print("Position embedding table shape:",
      position_embedding_table.shape)


# ==================================================
# 9. Get token embeddings
# ==================================================

x_embeddings = embedding_table[x]

print("x_embeddings shape:", x_embeddings.shape)


# ==================================================
# 10. Get position embeddings
# ==================================================

positions = torch.arange(block_size)

position_embeddings = position_embedding_table[positions]

print("position_embeddings shape:",
      position_embeddings.shape)


# ==================================================
# 11. Combine token + position embeddings
# ==================================================

x_embeddings_with_position = (
    x_embeddings + position_embeddings
)

print("combined embeddings shape:",
      x_embeddings_with_position.shape)


# ==================================================
# 12. Create linear layer
# ==================================================

linear_layer = torch.nn.Linear(
    embedding_dim,
    vocab_size
)

print("linear layer:", linear_layer)


# ==================================================
# 13. Make initial predictions
# ==================================================

model_logits = linear_layer(
    x_embeddings_with_position
)

print("model_logits shape:", model_logits.shape)


# ==================================================
# 14. Convert logits to probabilities
# ==================================================

model_probs = torch.softmax(
    model_logits,
    dim=1
)

print("model_probs shape:", model_probs.shape)


# ==================================================
# 15. Get probabilities for last position
# ==================================================

last_position_probs = model_probs[-1]

print("last position probabilities:",
      last_position_probs)

print("probability sum:",
      last_position_probs.sum())


# ==================================================
# 16. Sample a token
# ==================================================

sampled_id = torch.multinomial(
    last_position_probs,
    num_samples=1
)

sampled_character = id_to_char[
    sampled_id.item()
]

print("sampled character:",
      sampled_character)


# ==================================================
# 17. Calculate loss
# ==================================================

loss = torch.nn.functional.cross_entropy(
    model_logits,
    y
)

print("initial loss:", loss.item())


# ==================================================
# 18. Train the model
# ==================================================

learning_rate = 0.01

for step in range(100):

    # Make predictions
    model_logits = linear_layer(
        x_embeddings_with_position
    )

    # Calculate loss
    loss = torch.nn.functional.cross_entropy(
        model_logits,
        y
    )

    # Clear old gradients
    linear_layer.zero_grad()

    # Calculate new gradients
    loss.backward()

    # Update weights and bias
    with torch.no_grad():

        for parameter in linear_layer.parameters():
            parameter -= (
                learning_rate * parameter.grad
            )

    # Print progress
    if step % 10 == 0:
        print(
            "step:",
            step,
            "loss:",
            loss.item()
        )


# ==================================================
# 19. Check final loss
# ==================================================

model_logits = linear_layer(
    x_embeddings_with_position
)

final_loss = torch.nn.functional.cross_entropy(
    model_logits,
    y
)

print("\nfinal loss:", final_loss.item())


# ==================================================
# 20. Get model prediction
# ==================================================

last_position_logits = model_logits[-1]

predicted_id = torch.argmax(
    last_position_logits
)

predicted_character = id_to_char[
    predicted_id.item()
]

print("predicted character:",
      predicted_character)


# ==================================================
# 21. Get probability distribution
# ==================================================

last_position_probs = torch.softmax(
    last_position_logits,
    dim=0
)

print("last position probabilities:",
      last_position_probs)


# ==================================================
# 22. Sample from probability distribution
# ==================================================

sampled_id = torch.multinomial(
    last_position_probs,
    num_samples=1
)

sampled_character = id_to_char[
    sampled_id.item()
]

print("sampled character:",
      sampled_character)


# ==================================================
# 23. Generate text
# ==================================================

generated = x.clone()

for _ in range(10):

    # ----------------------------------------------
    # Keep only the latest block_size tokens
    # ----------------------------------------------

    context = generated[-block_size:]

    # ----------------------------------------------
    # Get token embeddings
    # ----------------------------------------------

    token_embeddings = embedding_table[context]

    # ----------------------------------------------
    # Get position embeddings
    # ----------------------------------------------

    positions = torch.arange(
        len(context)
    )

    position_embeddings = (
        position_embedding_table[positions]
    )

    # ----------------------------------------------
    # Combine token + position embeddings
    # ----------------------------------------------

    current_embeddings = (
        token_embeddings + position_embeddings
    )

    # ----------------------------------------------
    # Get model predictions
    # ----------------------------------------------

    model_logits = linear_layer(
        current_embeddings
    )

    # ----------------------------------------------
    # Get prediction for the last position
    # ----------------------------------------------

    last_position_logits = model_logits[-1]

    # ----------------------------------------------
    # Convert logits to probabilities
    # ----------------------------------------------

    last_position_probs = torch.softmax(
        last_position_logits,
        dim=0
    )

    # ----------------------------------------------
    # Sample next token
    # ----------------------------------------------

    sampled_id = torch.multinomial(
        last_position_probs,
        num_samples=1
    )

    # ----------------------------------------------
    # Add token to generated sequence
    # ----------------------------------------------

    generated = torch.cat(
        (generated, sampled_id)
    )


# ==================================================
# 24. Decode generated tokens
# ==================================================

generated_text = ""

for token_id in generated:

    generated_text += id_to_char[
        token_id.item()
    ]

print("\ngenerated token IDs:", generated)
print("generated text:", generated_text)
