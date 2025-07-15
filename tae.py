""" Excercises on Text Autoencoders"""
# %% [markdown]
# # Section 1: Text Autoencoders - Exploring SONAR
# ==============================================

# This notebook explores Meta's SONAR text autoencoder, which can encode text
# into fixed-size vectors and decode them back to (approximately) the original text.

# Learning objectives:
# 1. Load and use SONAR for text encoding/decoding
# 2. Understand the properties of text embeddings
# 3. Test robustness to noise
# 4. Explore how text length affects embeddings
# 5. Experiment with token swapping and sentence combinations

# %% [markdown]
# ## Setup and Installation
#
# First, we need to install SONAR and its dependencies. Just run, nothing worth reading here unless you get errors.
# Note: You may need to adjust the CUDA version in fairseq2 installation.

# %%
!pip install -q fairseq2==0.4.5 sonar-space==0.4.0 torchvision==0.21.0 torch==2.6.0 torchaudio==2.6.0 plotly nbformat

import torch
import numpy as np
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline
import torch.nn as nn
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import load_dataset
import json
from jaxtyping import Float

# Check if CUDA is available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE)
torch.set_grad_enabled(False)  # We're only doing inference
print(f"Using device: {DEVICE}")

# %% [markdown]
# ## Loading SONAR Models
#
# SONAR (Sentence-Level Multimodal and Language-Agnostic Representations) is Meta's text autoencoder
# that can encode entire sentences/paragraphs into fixed-size vectors and decode them back to approximately
# the original text.
#
# **What are Text Autoencoders?**
#
# Text Autoencoders are models that compress entire input sequences (sentences/paragraphs) into a single
# fixed-size vector representation (the "bottleneck"), then reconstruct the original text from that vector.
# Unlike typical text embedding models that only encode, these models have both an encoder AND decoder.
#
# ![Text Autoencoder Architecture](https://39669.cdn.cke-cs.com/rQvD3VnunXZu34m86e5f/images/db8d350884974ce6dcb1281011c5053e11b65711c12a4556.png)
#
# **How Text Autoencoders Work:**
# 1. **Encoder**: Takes input text → processes through Transformer → outputs single fixed-size vector (1024-dim)
# 2. **Bottleneck**: The compressed representation that captures semantic meaning in a dense vector
# 3. **Decoder**: Takes the vector → generates text that approximates the original input
#
# **Key Properties:**
# - **Lossy compression**: Some information is lost, but semantic meaning is preserved
# - **Fixed-size representation**: Any length text becomes same-size vector (useful for comparison/clustering)
# - **Cross-lingual**: Can encode in one language and decode in another
# - **Reconstruction capability**: Unlike embedding-only models, you can decode back to text
# - **Semantic preservation**: The bottleneck captures core meaning even with compression
#
# **SONAR Specifically:**
# - Trained on ~100B tokens with denoising and translation objectives
# - Uses 24-layer Transformer encoder and decoder, with mean-pooling to create the bottleneck vector
# - Supports 200+ languages and can handle up to 512 tokens of context
# - Currently one of the best-performing text autoencoders available
#

# %% [markdown]
# We start by loading the models.
print("Loading SONAR models...")
text2vec = TextToEmbeddingModelPipeline(
    encoder="text_sonar_basic_encoder",
    tokenizer="text_sonar_basic_encoder",
    device=DEVICE
)
vec2text = EmbeddingToTextModelPipeline(
    decoder="text_sonar_basic_decoder",
    tokenizer="text_sonar_basic_encoder",
    device=DEVICE
)
print("Models loaded successfully!")

# %% [markdown]
# ## Basic Usage - Encoding and Decoding
#
# Test basic encoding and decoding functionality.

# %%
# Simple example sentences
sentences = [
    'My name is SONAR.',
    'I can embed sentences into vectorial space.'
]

# Encode sentences to vectors
embeddings = text2vec.predict(sentences, source_lang="eng_Latn")
print(f"Embeddings shape: {embeddings.shape}")  # Should be [2, 1024]
print(f"Embedding dimension: {embeddings.shape[1]}")
print(f"L2 norm of embeddings: {torch.norm(embeddings, dim=1).tolist()}")

# Decode vectors back to text
reconstructed = vec2text.predict(embeddings, target_lang="eng_Latn", max_seq_len=512)
print("\nReconstruction quality:")
for orig, rec in zip(sentences, reconstructed):
    print(f"Original:      {orig}")
    print(f"Reconstructed: {rec}")
    print()

# %% [markdown]
# ## Exercise 1: Testing with Longer, More Realistic Text
# Let's test how well SONAR handles paragraph-length text.
#
# Write a function to reconstruct text from SONAR embeddings, and try testing with some longer text.

def reconstruct_text(texts: list[str]) -> list[str]:
    """Reconstruct text from SONAR embedding, by first encoding and then decoding the text.

    Args:
        texts: List of strings to reconstruct.

    Returns:
        List of reconstructed strings.
    """
    embedding = text2vec.predict(texts, source_lang="eng_Latn")
    return vec2text.predict(embedding, target_lang="eng_Latn", max_seq_len=512)

# Longer example paragraphs
paragraph1 = """SONAR is a model from August 2023, trained as a semantic text auto-encoder,
converting text into semantic embed vectors, which can later be decoded back into text.
Additionally, the model is trained such that the semantic embed vectors are to some degree
"universal" for different languages, and one can embed in French and decode in English."""

paragraph2 = """I tried it, and SONAR seems to work surprisingly well. For example, the above
paragraph and this paragraph, if each are encoded into two 1024 dimensional vectors
(one for each paragraph), the model returns the following decoded outputs."""

paragraph3 = """\
Your text here.
"""

# Test with paragraphs
long_texts = [paragraph1, paragraph2, paragraph3]
long_reconstructed = reconstruct_text(long_texts)

print("Paragraph reconstruction:")
for i, (orig, rec) in enumerate(zip(long_texts, long_reconstructed)):
    print(f"\n--- Paragraph {i+1} ---")
    print(f"Original ({len(orig)} chars):")
    print(orig[:100] + "..." if len(orig) > 100 else orig)
    print(f"\nReconstructed ({len(rec)} chars):")
    print(rec[:100] + "..." if len(rec) > 100 else rec)

# %% [markdown]
# How well does it work for longer text? It should be doing a pretty good job. Bonus: How long does the text get before you see some degradation?

# %% [markdown]
# ## Exercise 2: Noise Robustness Analysis
#
# In this exercise, we investigate SONAR's robustness to perturbations in the embedding space.
# We'll systematically add Gaussian noise of increasing magnitude to text embeddings and analyze
# how reconstruction quality degrades. This helps us understand:
# 1. How stable the embedding space is to small perturbations
# 2. The sensitivity of the decoder to different noise directions
#
# Write a function to test the robustness of SONAR to noise, and try it out with some different noise levels.

def test_noise_robustness(text, noise_levels):
    """Test how reconstruction quality degrades with noise.

    """
    # Get original embedding
    original_emb = text2vec.predict([text], source_lang="eng_Latn")
    original_norm = torch.norm(original_emb)

    results = []
    for noise_scale in noise_levels:
        # Add Gaussian noise
        noise = torch.randn_like(original_emb)
        noise = noise_scale * original_norm * noise / torch.norm(noise)
        noisy_emb = original_emb + noise

        # Decode noisy embedding
        reconstructed = vec2text.predict(noisy_emb, target_lang="eng_Latn", max_seq_len=512)[0]

        # Calculate cosine similarity
        cosine_sim = torch.nn.functional.cosine_similarity(
            original_emb, noisy_emb, dim=1
        ).item()

        results.append({
            'noise_scale': noise_scale,
            'cosine_similarity': cosine_sim,
            'reconstruction': reconstructed
        })

    return results

# Test with different noise levels
test_text = "The quick brown fox jumps over the lazy dog."
noise_levels = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]

print(f"Original text: {test_text}\n")
results = test_noise_robustness(test_text, noise_levels)

for res in results:
    print(f"Noise scale: {res['noise_scale']:.1f}")
    print(f"Cosine similarity: {res['cosine_similarity']:.3f}")
    print(f"Reconstructed: {res['reconstruction']}")
    print()

# %% [markdown]
# What do you see?
# It should be the case that with little noise, the reconstruction is still good. With more noise, the reconstruction gets worse. However, I found there is a lot of variance in the results, so try running it a few times. It seems like some directions have basically no effect, and others have a lot of effect.

# %% [markdown]
# ## Exercise 3: Text Length vs Vector Norm Analysis
#
# ### Exercise 3: Investigating the Relationship Between Text Length and Embedding Norms
#
# In this exercise, we'll explore whether there's a correlation between the length of text
# and the L2 norm (magnitude) of its embedding vector. This analysis will help us understand:
# - How semantic information is distributed across embedding dimensions
# - Whether longer texts result in larger embedding magnitudes
# - If the embedding space has inherent biases based on text length
#
# We'll test this hypothesis using three different types of text:
# 1. Repeated words (to test pure length effects)
# 2. Random character sequences (to test meaningless content)
# 3. Natural language sentences (to test realistic content)

# %%
import plotly.express as px
import pandas as pd
import random
import string

# Collect all data first
data = []
def add_data(text, text_type):
    emb = text2vec.predict([text], source_lang="eng_Latn")
    norm = torch.norm(emb).item()
    data.append({
        'text': text,
        'length': len(text),
        'norm': norm,
        'type': text_type
    })

# Repeated words (more examples)
for length in range(1, 100):
    for word in ['word', 'sentence', 'paragraph', 'dog', 'spicy', 'anime']:
        words = [word] * length
        text = ' '.join(words)
        add_data(text, 'Repeated Words')

# Random characters (more examples)
random.seed(42)
for length in range(1, 100, ):
        random_words = [''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 8))) for _ in range(length)]
        text = ' '.join(random_words)
        add_data(text, 'Random Characters')

# Normal sentences (many more examples)
normal_sentences = [
    "Hi",
    "Hello",
    "Good morning",
    "Hello there",
    "How are you?",
    "Nice to meet you",
    "The cat sat on the mat",
    "I like to read books",
    "The weather is nice today",
    "She went to the store yesterday",
    "The quick brown fox jumps over the lazy dog",
    "I enjoy listening to music in the evening",
    "She sells seashells by the seashore on weekends",
    "To be or not to be, that is the question",
    "The early bird catches the worm every morning",
    "A picture is worth a thousand words in most cases"
]
for text in normal_sentences:
    add_data(text, 'Real Text')


dataset = load_dataset("nickypro/fineweb-llama3b-regen-split", split="train")
for split_text in dataset.select(range(20)):
    for paragraph in split_text['split_text']:
        add_data(paragraph, 'Real Text')


# Create DataFrame and plot
df = pd.DataFrame(data)
# Truncate text to first 50 characters for hover display
df['text_truncated'] = df['text'].str[:50] + '...'
fig = px.scatter(df,
        x='length', y='norm', color='type',
        title="Text Length vs Embedding Norm",
        labels={'length': 'Text Length (characters)', 'norm': 'Embedding L2 Norm'},
        hover_data=['text_truncated'],
        opacity=0.5,
        log_x=True)

fig.show()

# %% [markdown]
# ## Exercise 4: Token Swapping Experiments
#
# This exercise explores how we can manipulate text embeddings to perform token swapping.
# We'll investigate:
# 1. Building difference vectors between similar texts
# 2. Applying global transformations to swap words
# 3. Creating position-specific transformations for targeted edits

# %%
# Helper functions

def diff_vector(src_text: str, tgt_text: str) -> torch.Tensor:
    """Return embedding difference between *tgt_text* and *src_text* (tgt − src)."""

    src_emb = text2vec.predict([src_text], source_lang="eng_Latn")
    tgt_emb = text2vec.predict([tgt_text], source_lang="eng_Latn")
    return (tgt_emb - src_emb).squeeze(0)

def decode(embedding: torch.Tensor) -> str:
    """Greedy‑decode a single 1024‑D embedding back to text."""
    return vec2text.predict(embedding.unsqueeze(0), target_lang="eng_Latn", max_seq_len=512)[0]


def positional_diff(src_word: str, tgt_word: str, pos: int, *, seq_len: int, filler: str = "_") -> torch.Tensor:
    """Build a difference vector that swaps **src_word→tgt_word** at index *pos*.

    All other positions are filled with *filler* tokens so that the vector is
    specific to that location.
    """
    src_tokens = [filler] * seq_len
    tgt_tokens = [filler] * seq_len
    src_tokens[pos] = src_word
    tgt_tokens[pos] = tgt_word
    return diff_vector(" ".join(src_tokens), " ".join(tgt_tokens))

# %% [markdown]
# Now we can try see what the difference vector does in different cases.

# 1. Global dog→cat vector
print("1. Global word swapping:")
swap_vec = diff_vector("dog", "cat")
sentence = "the dog is happy in the dog house"
sent_emb = text2vec.predict([sentence], source_lang="eng_Latn").squeeze(0)

print(f"Original:               {decode(sent_emb)}")
print(f"Global swap dog→cat:    {decode(sent_emb + swap_vec)}")

# 2. Position‑specific swap
print("\n2. Position-specific swapping:")
# Swap only the token at index 1 (0‑based) in a sentence
pos_vec = positional_diff("dog", "cat", pos=1, seq_len=8, filler="a")
print(f"Position‑aware swap:    {decode(sent_emb + pos_vec)}")

# 3. Test with different word pairs
print("\n3. Testing different word pairs:")
word_pairs = [("happy", "sad"), ("house", "tree"), ("big", "small")]
for src, tgt in word_pairs:
    swap_vec = diff_vector(src, tgt)
    test_sentence = f"the {src} animal lives here"
    test_emb = text2vec.predict([test_sentence], source_lang="eng_Latn").squeeze(0)
    print(f"{src}→{tgt}: '{test_sentence}' → '{decode(test_emb + swap_vec)}'")


# %% [markdown]
# ## Exercise 7: Sentence Combination
#
# This exercise explores how we can combine two sentences into a single embedding.
# So far I have only tried a couple of the most naive approaches. It's ok but I suspect it should be easy to try better approaches to this also.

# %% [markdown]
# ### Part 1: Basic Combination Analysis
#
# First, let's analyze how SONAR combines sentences with different relationships.

# %%
# Create diverse sentence pairs for analysis
sentence_pairs = [
    # Related sentences (continuation)
    ("The weather is beautiful today", "I think I'll go for a walk"),
    ("She opened the mysterious letter", "Her hands trembled as she read it"),

    # Contrasting sentences
    ("I love sunny days", "But I hate the rain"),
    ("The movie was exciting", "However, the ending disappointed me"),

    # Unrelated sentences
    ("Cats are independent animals", "Python is a programming language"),
    ("The Earth orbits the Sun", "Pizza is my favorite food"),

    # Question-answer pairs
    ("What's your favorite color?", "My favorite color is blue"),
    ("Where do you live?", "I live in New York City"),
]

# Analyze combinations
combination_data = []
for sent_a, sent_b in sentence_pairs:
    # Individual embeddings
    emb_a = text2vec.predict([sent_a], source_lang="eng_Latn")
    emb_b = text2vec.predict([sent_b], source_lang="eng_Latn")

    # Combined embeddings (both orders)
    combined_ab = f"{sent_a} {sent_b}"
    combined_ba = f"{sent_b} {sent_a}"
    emb_ab = text2vec.predict([combined_ab], source_lang="eng_Latn")
    emb_ba = text2vec.predict([combined_ba], source_lang="eng_Latn")

    # Various combinations
    emb_avg = (emb_a + emb_b) / 2
    emb_sum = emb_a + emb_b
    emb_diff = emb_a - emb_b

    # Calculate similarities
    data = {
        'sent_a': sent_a[:30] + '...' if len(sent_a) > 30 else sent_a,
        'sent_b': sent_b[:30] + '...' if len(sent_b) > 30 else sent_b,
        'sim_ab_a': torch.nn.functional.cosine_similarity(emb_ab, emb_a).item(),
        'sim_ab_b': torch.nn.functional.cosine_similarity(emb_ab, emb_b).item(),
        'sim_ab_ba': torch.nn.functional.cosine_similarity(emb_ab, emb_ba).item(),
        'sim_ab_avg': torch.nn.functional.cosine_similarity(emb_ab, emb_avg).item(),
        'sim_ab_sum': torch.nn.functional.cosine_similarity(emb_ab, emb_sum).item(),
        'order_sensitivity': torch.norm(emb_ab - emb_ba).item()
    }
    combination_data.append(data)

# Display results
df_comb = pd.DataFrame(combination_data)
print("Sentence Combination Analysis:")
print(df_comb.to_string(index=False))

# %% [markdown]
# ### Part 2: Generate Training Data
#
# We'll create a dataset of sentence pairs and their combined embeddings to train our model.

# %%
# Load a dataset for diverse sentences
from datasets import load_dataset

print("Generating training data...")
dataset = load_dataset("nickypro/fineweb-llama3b-regen-split", split="train")

# Extract individual sentences
all_sentences = []
for item in dataset.select(range(100)):  # Use first 100 documents
    for paragraph in item['split_text']:
        # Split paragraph into sentences (simple approach)
        sentences = paragraph.split('. ')
        for sent in sentences:
            if 10 < len(sent) < 200:  # Filter by length
                all_sentences.append(sent.strip())

# Limit to manageable size
all_sentences = all_sentences[:2000]
print(f"Collected {len(all_sentences)} sentences")

# Create pairs and compute embeddings
print("Creating sentence pairs and embeddings...")
training_data = []
n_pairs = 1000  # Number of pairs to create

for i in tqdm(range(n_pairs)):
    # Randomly select two sentences
    idx1, idx2 = np.random.choice(len(all_sentences), 2, replace=False)
    sent1, sent2 = all_sentences[idx1], all_sentences[idx2]

    # Compute embeddings
    emb1 = text2vec.predict([sent1], source_lang="eng_Latn")
    emb2 = text2vec.predict([sent2], source_lang="eng_Latn")

    # Compute combined embedding
    combined = f"{sent1} {sent2}"
    emb_combined = text2vec.predict([combined], source_lang="eng_Latn")

    training_data.append({
        'emb1': emb1.cpu(),
        'emb2': emb2.cpu(),
        'emb_combined': emb_combined.cpu(),
        'sent1': sent1,
        'sent2': sent2
    })

print(f"Generated {len(training_data)} training examples")


# %% [markdown]
# ### Part 3: Try simple linear combination
# If we want to combine two sentences, we can just add their embeddings? Or maybe average them? Will this give us something that works as an embedding with two sentences side-by-side?

# %%
class SimpleLinearCombiner(nn.Module):
    def __init__(self, embed_dim=1024):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x, y):
        return x + y

basic_combiner_model = SimpleLinearCombiner().to(DEVICE)

# Test the simple linear combiner
def test_performance_on_new_examples(model, verbose=True):
    """Test model performance on predefined pairs plus one random example"""
    model.eval()

    # Predefined test pairs
    test_pairs = [
        ("It started raining heavily.", "Everyone ran for shelter."),
        ("First, preheat the oven.", "Then, mix the ingredients."),
        ("The book was fascinating.", "The movie adaptation was terrible."),
        ("I need to buy milk.", "I also need to get bread."),
    ]

    # Add one random pair
    idx1, idx2 = np.random.choice(len(all_sentences), 2, replace=False)
    test_pairs.append((all_sentences[idx1], all_sentences[idx2]))

    test_results = []

    for sent1, sent2 in test_pairs:
        # Get embeddings
        emb1 = text2vec.predict([sent1], source_lang="eng_Latn").to(DEVICE)
        emb2 = text2vec.predict([sent2], source_lang="eng_Latn").to(DEVICE)
        emb_true = text2vec.predict([f"{sent1} {sent2}"], source_lang="eng_Latn").to(DEVICE)

        # Predict and decode
        with torch.no_grad():
            emb_pred = model(emb1.squeeze(0), emb2.squeeze(0)).unsqueeze(0)

        text_true = vec2text.predict(emb_true.cpu(), target_lang="eng_Latn")[0]
        text_pred = vec2text.predict(emb_pred.cpu(), target_lang="eng_Latn")[0]
        similarity = torch.cosine_similarity(emb_pred, emb_true, dim=-1).item()

        test_results.append({
            'sent1': sent1, 'sent2': sent2, 'decoded_true': text_true,
            'decoded_pred': text_pred, 'similarity': similarity
        })

        if verbose:
            print(f"\nSent1: {sent1}")
            print(f"Sent2: {sent2}")
            print(f"True: {text_true}")
            print(f"Pred: {text_pred}")
            print(f"Similarity: {similarity:.4f}")

    avg_similarity = np.mean([r['similarity'] for r in test_results])
    print(f"\nAverage similarity: {avg_similarity:.4f}")
    return test_results

# Test the simple linear combiner
test_results = test_performance_on_new_examples(basic_combiner_model)


# %% [markdown]
# What do you see?
# In general, you should see that this kinda gets a sentence that is the same as one of the original sentences, or inbetween the two sentences. It doesn't really append one sentence to the other.

# %% [markdown]
# ### Part 4: Trained scale combination model
#
# We'll create simple model that learns to combine two sentence embeddings.

# %%
class ScaleCombinerModel(nn.Module):
    """
    Simple linear combiner model:
    output = const + (scale1)*x + (scale2)*y
    """
    def __init__(self, embed_dim=1024):
        super().__init__()
        self.embed_dim = embed_dim

        # Constant bias
        self.const = nn.Parameter(torch.zeros(embed_dim))

        # Scalar weights for original embeddings
        self.scale1 = nn.Parameter(torch.ones(1) * 0.5)
        self.scale2 = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x, y):
        # Simple linear combination
        output = self.const + self.scale1 * x + self.scale2 * y
        return output

# Initialize model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scale_combiner_model = ScaleCombinerModel(embed_dim=1024).to(DEVICE)
print(f"Model parameters: {sum(p.numel() for p in scale_combiner_model.parameters()):,}")

# %% [markdown]
# ### Part 5: Train the Model

# %%

def train_combiner_model(model, training_data):
    """Train the combiner model on the provided training data."""
    torch.set_grad_enabled(True)  # We're now training

    # Prepare data for training
    X1 = torch.stack([d['emb1'].squeeze(0) for d in training_data])
    X2 = torch.stack([d['emb2'].squeeze(0) for d in training_data])
    Y = torch.stack([d['emb_combined'].squeeze(0) for d in training_data])

    # Split into train/test
    X1_train, X1_test, X2_train, X2_test, Y_train, Y_test = train_test_split(
        X1, X2, Y, test_size=0.2, random_state=42
    )
    X1_train = X1_train.to(DEVICE)
    X2_train = X2_train.to(DEVICE)
    Y_train = Y_train.to(DEVICE)
    X1_test = X1_test.to(DEVICE)
    X2_test = X2_test.to(DEVICE)
    Y_test = Y_test.to(DEVICE)

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Training loop
    epochs = 10
    batch_size = 32
    train_losses = []
    test_losses = []

    print("Training a combiner model...")
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0
        for i in range(0, len(X1_train), batch_size):
            batch_x1 = X1_train[i:i+batch_size]
            batch_x2 = X2_train[i:i+batch_size]
            batch_y = Y_train[i:i+batch_size]

            optimizer.zero_grad()
            pred = model(batch_x1, batch_x2)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Evaluation
        model.eval()
        with torch.no_grad():
            train_pred = model(X1_train, X2_train)
            train_loss = criterion(train_pred, Y_train).item()

            test_pred = model(X1_test, X2_test)
            test_loss = criterion(test_pred, Y_test).item()

            train_losses.append(train_loss)
            test_losses.append(test_loss)

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}")

    return train_losses, test_losses

# Train the model
try:
    torch.set_grad_enabled(True)  # We're now training but only in this cell
    train_losses, test_losses = train_combiner_model(scale_combiner_model, training_data)
except Exception as e:
    print(f"Error training model: {e}")
    print(e.traceback)
finally:
    torch.set_grad_enabled(False)  #

# %% [markdown]
# ### Part 6: Test Performance on New Examples

# %%
print("\nModel Performance on Test Examples:")
print("=" * 80)

test_results = test_performance_on_new_examples(scale_combiner_model)

# %% [markdown]
# What do you see?
# It does a better job, it seems to be approximately one sentence followed by the other, but kind still mixes the two sentences up a but sometimes.

# %%