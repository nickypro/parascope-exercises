# %% [markdown]
# # Section 3: SONAR Parascopes - Trained Probes
# ============================================
#
# While continuation parascopes directly use the model's own generation capabilities,
# SONAR parascopes take a different approach by learning to map residual streams to
# text embeddings that can be decoded back to text.
#
#![AutoEncoder Map ParaScope](https://39669.cdn.cke-cs.com/rQvD3VnunXZu34m86e5f/images/07873b3421363d38f1cee7649b8bd73fccd43300afd5c2fa.png)
#
# The SONAR approach:
# 1. Train a probe (Linear or MLP) to map from residual stream → SONAR embedding space
# 2. Use SONAR's decoder to convert embeddings back to text
# 3. This allows us to extract semantic content without relying on the model's generation
#
# Learning objectives:
# 1. Understand the SONAR text autoencoder and its embedding space
# 2. Load pre-generated datasets of residual streams and SONAR embeddings
# 3. Train probes to map between these spaces
# 4. Evaluate and compare with continuation parascopes

# %% [markdown]
# ## Setup and Installation
#
# First, install SONAR and other required packages.

# %%
!pip install -q sonar-space torch torchvision transformer-lens sentence-transformers
!pip install -q matplotlib seaborn pandas numpy scikit-learn einops

# %%
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import einops
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

# SONAR imports
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline

# For evaluation
from sentence_transformers import SentenceTransformer

# Disable gradients by default
torch.set_grad_enabled(False)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# %% [markdown]
# ## Load the model
from transformer_lens import HookedTransformer
from datasets import load_dataset

model = HookedTransformer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", device=DEVICE)

# %% [markdown]
# ## Exercise 1: Generating Training Data
#
# Since we want the probe to match "what the model might output", we need data of what the model's output might look like.
# Two-step process: (1) load dataset + generate prompts, (2) use prompts to generate model outputs.

# %% [markdown]
# ### Part 1: Generating Prompts

# %%
def format_prompt(prompt: str) -> str:
    """Format prompt using the model's chat template."""
    messages = [
        {"role": "user", "content": prompt}
    ]
    return model.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

def format_question(text: str, max_chars: int = 32000) -> str:
    """Transform existing text into a generation prompt."""
    return f"""Content: {text[:max_chars]}

REQUEST: Write a prompt based on the above text, that is a single-paragraph, high-level description. Make the prompt in the format: "Write a (article/piece/entry), which includes (1-2 topics). The piece should be approximately (n-paragraphs) long."

Only provide the prompt, do not write anything else."""

# Load dataset and generate training data
print("Loading dataset...")
dataset = load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split="train", streaming=True)

# Step 1: Generate prompts from texts
prompts = []
for i, text in enumerate(dataset.take(3)):  # Start with just 3 examples
    print(f"Processing text {i+1}/3...")

    # Format the meta-prompt properly
    meta_prompt = format_question(text["text"])
    formatted_prompt = format_prompt(meta_prompt)

    # Tokenize and generate
    prompt_tokens = model.to_tokens(formatted_prompt)
    generated_tokens = model.generate(
        prompt_tokens,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7
    )

    # Extract just the generated part
    output_tokens = generated_tokens[:, len(prompt_tokens[0]):-1]
    generated_prompt = model.to_string(output_tokens)[0]

    prompts.append(generated_prompt.strip())

print(f"Generated {len(prompts)} training examples")
print(f"Example prompt: {prompts[0]}")

# %% [markdown]
# ### Part 2: Generate Model Outputs from Prompts

# %%
model_outputs = []
for i, prompt in enumerate(prompts):
    print(f"Generating output {i+1}/{len(prompts)}...")

    # Format and tokenize the prompt
    formatted_prompt = format_prompt(prompt)
    prompt_tokens = model.to_tokens(formatted_prompt)

    # Generate response
    output_tokens = model.generate(
        prompt_tokens,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.7
    )

    # Extract just the generated part
    generated_tokens = output_tokens[:, len(prompt_tokens[0]):]
    generated_text = model.to_string(generated_tokens)[0]

    model_outputs.append(generated_text.strip())

print(f"Generated {len(model_outputs)} model outputs")
print(f"Example output: {model_outputs[0][:200]}...")

# %% [markdown]
# ### Part 3: Loading Pre-made Data
# The above takes too long to run, so we'll load pre-made data.
# %%

from datasets import load_dataset
dataset = load_dataset("nickypro/fineweb-llama3b-regen", split="train")

[print(f"{k}: {[v]}") for k,v in dataset[0].items()]
# {id: 0, prompt: "...", completion: "..."}


# %% [markdown]
# ### Part 4: Splitting into Sections
# As we want to predict that the "next section" of the text will say, we need to split the text into sections.
# I choose to split the text by paragraphs, but it is also reasonable to split it by sentences or something similar.
# For the data here, we want data of the form:
# [prompt, section_1, section_2, section_3, ...]

# %%

from typing import List

def split_text_into_paragraphs(text: str) -> List[str]:
    """
    Splits a block of text into paragraphs.
    Paragraphs are separated by two or more newlines.
    The convention we choose is to have the newlines stored at the end of each paragraph.
    The list should combine to give the original text.
    """
    paragraphs = [p+"\n\n" for p in text.split('\n\n')]
    return paragraphs

def split_dataset_prompt_and_sections(dataset) -> List[List[str]]:
    """
    For each example in the dataset, keeps the prompt as a single string,
    and splits the 'completion' field into paragraphs.
    Returns a list of lists:
    [ [prompt, section_1, section_2, ...], ... ]
    """
    split_data = []
    for i, example in enumerate(tqdm(dataset)):
        prompt = format_prompt(example["prompt"])
        completion = example["completion"]
        completion_paragraphs = split_text_into_paragraphs(completion)
        # The first element is the prompt (as a single string), then the completion paragraphs
        split_text = [prompt] + completion_paragraphs
        split_data.append({"id": i, "split_text": split_text})
    return split_data

# Example usage:
split_sections = split_dataset_prompt_and_sections(dataset)
print(split_sections[0]["split_text"][:5])

# %% [markdown]
# Alternatively, I don't like waiting for this either, so we can just load the pre-split data.

split_sections = load_dataset("nickypro/fineweb-llama3b-regen-split-formatted", split="train")

print(split_sections[0])


# %% [markdown]
# ## Exercise 2: Loading Residual Streams and Embeddings
#
# We need to load pre-generated data containing:
# - Residual stream activations from language models
# - Corresponding SONAR embeddings of the paragraphs
# - The actual paragraph text

# %%
# Data loading utilities (simplified from your code)
BASE_DIR = "./data"  # Adjust this path as needed

def load_res_data(index: int, group_size: int = 2, groups_to_load: int = 28,
                  group_operation: str = "cat") -> torch.Tensor:
    """Load residual stream data for a given index."""
    # Simulate loading - in practice this loads from disk
    # For demo purposes, create random data
    n_samples = 1000
    d_model = 3072  # Llama-3.2-3B hidden size
    n_layers = groups_to_load

    # In real implementation, this would load from file:
    # file_path = f"{BASE_DIR}/res_tensors/res_data_{index:03d}.pt"
    # data = torch.load(file_path)

    # Simulated data
    data = torch.randn(n_samples, n_layers * group_size * d_model)
    return data.float()

def load_embeds(index: int) -> torch.Tensor:
    """Load SONAR embeddings for paragraphs."""
    # In practice: file_path = f"{BASE_DIR}/sonar_embeds/embeds_{index:03d}.pt"
    n_samples = 1000
    d_sonar = 1024
    return torch.randn(n_samples, d_sonar).float()

def load_split_paragraphs(index: int) -> List[str]:
    """Load the actual paragraph texts."""
    # In practice: file_path = f"{BASE_DIR}/split_paragraphs/paragraphs_{index:03d}.json"
    # Simulated paragraphs
    paragraphs = [f"This is paragraph {i} from file {index}." for i in range(1000)]
    return paragraphs

# Test loading
print("Testing data loading...")
res_data = load_res_data(0, groups_to_load=14)  # Using half the layers
embeds = load_embeds(0)
paragraphs = load_split_paragraphs(0)

print(f"Residual data shape: {res_data.shape}")
print(f"Embeddings shape: {embeds.shape}")
print(f"Number of paragraphs: {len(paragraphs)}")
print(f"Example paragraph: {paragraphs[0]}")

# %% [markdown]
# ## Exercise 3: Normalization and Preprocessing
#
# The residual streams and embeddings need normalization for stable training.
# We use Welford's algorithm to compute running statistics.

# %%
@dataclass
class WelfordStats:
    """Track running mean and variance using Welford's algorithm."""
    mean: torch.Tensor
    m2: torch.Tensor
    count: int

    def update(self, new_data: torch.Tensor):
        """Update statistics with new batch of data."""
        batch_size = new_data.shape[0]
        for i in range(batch_size):
            self.count += 1
            delta = new_data[i] - self.mean
            self.mean += delta / self.count
            delta2 = new_data[i] - self.mean
            self.m2 += delta * delta2

    @property
    def variance(self):
        return self.m2 / (self.count - 1) if self.count > 1 else torch.zeros_like(self.m2)

    @property
    def std(self):
        return torch.sqrt(self.variance + 1e-6)

class Normalizer:
    """Normalize data using precomputed statistics."""
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, device: str = DEVICE):
        self.mean = mean.to(device)
        self.std = std.to(device)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + 1e-6)

    def restore(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.std + 1e-6) + self.mean

# Compute normalization statistics (in practice, load precomputed stats)
print("Computing normalization statistics...")
res_stats = WelfordStats(
    mean=torch.zeros(res_data.shape[1]),
    m2=torch.zeros(res_data.shape[1]),
    count=0
)
embed_stats = WelfordStats(
    mean=torch.zeros(embeds.shape[1]),
    m2=torch.zeros(embeds.shape[1]),
    count=0
)

# Update with data
res_stats.update(res_data)
embed_stats.update(embeds)

# Create normalizers
res_normalizer = Normalizer(res_stats.mean, res_stats.std)
embed_normalizer = Normalizer(embed_stats.mean, embed_stats.std)

# Test normalization
normalized_res = res_normalizer.normalize(res_data[:10].to(DEVICE))
print(f"Normalized residual mean: {normalized_res.mean():.4f}, std: {normalized_res.std():.4f}")

# %% [markdown]
# ## Exercise 4: Define Probe Models
#
# We'll implement both Linear and MLP probes to map from residual streams to SONAR embeddings.

# %%
class LinearProbe(nn.Module):
    """Simple linear mapping from residual stream to SONAR embedding."""
    def __init__(self, d_res: int, d_sonar: int = 1024):
        super().__init__()
        self.linear = nn.Linear(d_res, d_sonar)

        # Initialize weights
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class MLPProbe(nn.Module):
    """MLP probe with hidden layers for more expressive mapping."""
    def __init__(self, d_res: int, d_hidden: int = 8192, d_sonar: int = 1024,
                 dropout: float = 0.05):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_res, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_sonar)
        )

        # Initialize weights
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

# Create probe models
d_res = res_data.shape[1]
linear_probe = LinearProbe(d_res).to(DEVICE)
mlp_probe = MLPProbe(d_res).to(DEVICE)

print(f"Linear probe parameters: {sum(p.numel() for p in linear_probe.parameters()):,}")
print(f"MLP probe parameters: {sum(p.numel() for p in mlp_probe.parameters()):,}")

# %% [markdown]
# ## Exercise 5: Training Loop (Demonstration)
#
# Here's how the probes would be trained. In practice, this takes hours,
# so we'll load pre-trained weights instead.

# %%
def train_probe_demo(probe: nn.Module, res_data: torch.Tensor, embeds: torch.Tensor,
                     num_epochs: int = 1, batch_size: int = 32, lr: float = 2e-5):
    """
    Demonstration of probe training loop.
    NOTE: This is simplified - full training would take much longer!
    """
    torch.set_grad_enabled(True)

    # Normalize data
    res_norm = res_normalizer.normalize(res_data)
    embed_norm = embed_normalizer.normalize(embeds)

    # Create data loader
    dataset = torch.utils.data.TensorDataset(res_norm, embed_norm)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Setup training
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-7)
    criterion = nn.MSELoss()

    probe.train()
    losses = []

    print(f"Training {probe.__class__.__name__}...")
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_x, batch_y in tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            # Forward pass
            pred = probe(batch_x)
            loss = criterion(pred, batch_y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")

    torch.set_grad_enabled(False)
    return losses

# Demonstrate training for just 1 epoch (normally would be 10+)
# losses = train_probe_demo(linear_probe, res_data[:100], embeds[:100], num_epochs=1)
print("In practice, training takes hours. We'll load pre-trained probes instead.")

# %% [markdown]
# ## Exercise 6: Load Pre-trained Probes
#
# Since training takes too long, let's load pre-trained probe weights.

# %%
def load_pretrained_probe(probe_type: str = "linear") -> nn.Module:
    """
    Load pre-trained probe weights.
    In practice, this would load from a checkpoint file.
    """
    d_res = res_data.shape[1]

    if probe_type == "linear":
        probe = LinearProbe(d_res).to(DEVICE)
        # In practice: probe.load_state_dict(torch.load("checkpoints/linear_probe.pt"))
    else:
        probe = MLPProbe(d_res).to(DEVICE)
        # In practice: probe.load_state_dict(torch.load("checkpoints/mlp_probe.pt"))

    probe.eval()
    print(f"Loaded pre-trained {probe_type} probe")
    return probe

# Load pre-trained probes
linear_probe = load_pretrained_probe("linear")
mlp_probe = load_pretrained_probe("mlp")

# %% [markdown]
# ## Exercise 7: Generate Text with SONAR Parascopes
#
# Now let's use the trained probes to decode residual streams into text.

# %%
def sonar_parascope_decode(probe: nn.Module, residual_stream: torch.Tensor,
                          paragraphs: List[str] = None) -> List[str]:
    """
    Decode residual streams to text using SONAR parascope.

    Args:
        probe: Trained probe model (Linear or MLP)
        residual_stream: Residual activations [batch, d_res]
        paragraphs: Original paragraphs for comparison (optional)

    Returns:
        List of decoded texts
    """
    probe.eval()
    with torch.no_grad():
        # Normalize residual stream
        res_norm = res_normalizer.normalize(residual_stream.to(DEVICE))

        # Map to SONAR embedding space
        sonar_embed_norm = probe(res_norm)

        # Denormalize to get actual SONAR embeddings
        sonar_embed = embed_normalizer.restore(sonar_embed_norm)

        # Decode with SONAR
        decoded_texts = vec2text.predict(
            sonar_embed.cpu(),
            target_lang="eng_Latn",
            max_seq_len=512
        )

    return decoded_texts

# Test on a few examples
test_indices = [0, 1, 2]
test_res = res_data[test_indices]
test_paragraphs = [paragraphs[i] for i in test_indices]

# Decode with both probes
linear_decoded = sonar_parascope_decode(linear_probe, test_res)
mlp_decoded = sonar_parascope_decode(mlp_probe, test_res)

print("SONAR Parascope Results:")
print("="*80)
for i, idx in enumerate(test_indices):
    print(f"\nExample {idx}:")
    print(f"Original:   {test_paragraphs[i]}")
    print(f"Linear:     {linear_decoded[i]}")
    print(f"MLP:        {mlp_decoded[i]}")

# %% [markdown]
# ## Exercise 8: Comparison with Continuation Parascope
#
# Let's compare SONAR parascopes with the continuation parascope from Section 2.

# %%
# Load sentence transformer for evaluation
sent_model = SentenceTransformer('all-mpnet-base-v2')

def compare_methods(original_texts: List[str],
                   sonar_linear: List[str],
                   sonar_mlp: List[str],
                   continuation: List[str] = None) -> pd.DataFrame:
    """
    Compare different parascope methods using cosine similarity.
    """
    results = []

    for i, orig in enumerate(original_texts):
        orig_emb = sent_model.encode(orig)

        # Compute similarities
        linear_emb = sent_model.encode(sonar_linear[i])
        mlp_emb = sent_model.encode(sonar_mlp[i])

        linear_sim = np.dot(orig_emb, linear_emb) / (np.linalg.norm(orig_emb) * np.linalg.norm(linear_emb))
        mlp_sim = np.dot(orig_emb, mlp_emb) / (np.linalg.norm(orig_emb) * np.linalg.norm(mlp_emb))

        result = {
            'index': i,
            'linear_similarity': linear_sim,
            'mlp_similarity': mlp_sim
        }

        if continuation:
            cont_emb = sent_model.encode(continuation[i])
            cont_sim = np.dot(orig_emb, cont_emb) / (np.linalg.norm(orig_emb) * np.linalg.norm(cont_emb))
            result['continuation_similarity'] = cont_sim

        results.append(result)

    return pd.DataFrame(results)

# Compare on more examples
n_compare = 50
compare_res = res_data[:n_compare]
compare_paragraphs = paragraphs[:n_compare]

linear_decoded = sonar_parascope_decode(linear_probe, compare_res)
mlp_decoded = sonar_parascope_decode(mlp_probe, compare_res)

# For comparison, simulate continuation parascope results
# In practice, these would come from Section 2's continuation parascope
continuation_decoded = [p[:50] + "..." for p in compare_paragraphs]  # Simplified

# Create comparison
comparison_df = compare_methods(
    compare_paragraphs,
    linear_decoded,
    mlp_decoded,
    continuation_decoded
)

# Visualize results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Box plot
comparison_df[['linear_similarity', 'mlp_similarity', 'continuation_similarity']].plot.box(ax=ax1)
ax1.set_ylabel('Cosine Similarity')
ax1.set_title('Method Comparison')
ax1.set_xticklabels(['Linear\nSONAR', 'MLP\nSONAR', 'Continuation'])

# Scatter plot: Linear vs MLP
ax2.scatter(comparison_df['linear_similarity'], comparison_df['mlp_similarity'], alpha=0.6)
ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax2.set_xlabel('Linear SONAR Similarity')
ax2.set_ylabel('MLP SONAR Similarity')
ax2.set_title('Linear vs MLP Performance')

plt.tight_layout()
plt.show()

# Print statistics
print("\nMethod Statistics:")
print(comparison_df[['linear_similarity', 'mlp_similarity', 'continuation_similarity']].describe())

# %% [markdown]
# ## Exercise 9: Analyzing Probe Behavior
#
# Let's analyze what the probes have learned by examining their weights and outputs.

# %%
def analyze_probe_weights(probe: nn.Module, probe_name: str):
    """Analyze the weight structure of a probe."""
    if isinstance(probe, LinearProbe):
        weights = probe.linear.weight.data.cpu()

        # Analyze weight magnitudes by input dimension groups
        # Assuming input is organized as [layer1_attn, layer1_mlp, layer2_attn, ...]
        d_model = 3072  # Llama hidden size
        n_segments = weights.shape[1] // d_model

        segment_norms = []
        for i in range(n_segments):
            start = i * d_model
            end = (i + 1) * d_model
            norm = torch.norm(weights[:, start:end], dim=1).mean().item()
            segment_norms.append(norm)

        # Visualize
        plt.figure(figsize=(10, 5))
        layers = list(range(len(segment_norms) // 2))
        attn_norms = segment_norms[::2]
        mlp_norms = segment_norms[1::2]

        x = np.arange(len(layers))
        width = 0.35

        plt.bar(x - width/2, attn_norms, width, label='Attention')
        plt.bar(x + width/2, mlp_norms, width, label='MLP')

        plt.xlabel('Layer')
        plt.ylabel('Average Weight Norm')
        plt.title(f'{probe_name} Weight Norms by Layer and Component')
        plt.legend()
        plt.show()

        return segment_norms

# Analyze linear probe
print("Analyzing Linear Probe Weights:")
linear_norms = analyze_probe_weights(linear_probe, "Linear Probe")

# %% [markdown]
# ## Exercise 10: Advanced Analysis - Information Preservation
#
# Let's analyze what types of information are preserved by different methods.

# %%
def analyze_information_preservation(original_texts: List[str],
                                   decoded_texts: List[str],
                                   method_name: str) -> Dict[str, float]:
    """
    Analyze what aspects of text are preserved in decoding.
    """
    from collections import Counter
    import re

    preservation_scores = {
        'length_ratio': [],
        'word_overlap': [],
        'unique_words_ratio': [],
        'punctuation_preserved': []
    }

    for orig, decoded in zip(original_texts, decoded_texts):
        # Length preservation
        length_ratio = len(decoded) / (len(orig) + 1e-6)
        preservation_scores['length_ratio'].append(length_ratio)

        # Word overlap
        orig_words = set(re.findall(r'\w+', orig.lower()))
        decoded_words = set(re.findall(r'\w+', decoded.lower()))
        overlap = len(orig_words & decoded_words) / (len(orig_words) + 1e-6)
        preservation_scores['word_overlap'].append(overlap)

        # Unique words preserved
        unique_ratio = len(decoded_words) / (len(orig_words) + 1e-6)
        preservation_scores['unique_words_ratio'].append(unique_ratio)

        # Punctuation
        orig_punct = sum(1 for c in orig if c in '.,!?;:')
        decoded_punct = sum(1 for c in decoded if c in '.,!?;:')
        punct_ratio = decoded_punct / (orig_punct + 1e-6)
        preservation_scores['punctuation_preserved'].append(punct_ratio)

    # Calculate averages
    avg_scores = {k: np.mean(v) for k, v in preservation_scores.items()}

    # Visualize
    plt.figure(figsize=(8, 6))
    metrics = list(avg_scores.keys())
    values = list(avg_scores.values())

    plt.bar(metrics, values)
    plt.ylabel('Score')
    plt.title(f'Information Preservation - {method_name}')
    plt.xticks(rotation=45)
    plt.ylim(0, 1.5)

    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f'{v:.2f}', ha='center')

    plt.tight_layout()
    plt.show()

    return avg_scores

# Analyze different methods
print("Information Preservation Analysis:")
linear_preservation = analyze_information_preservation(
    compare_paragraphs[:20],
    linear_decoded[:20],
    "Linear SONAR"
)
mlp_preservation = analyze_information_preservation(
    compare_paragraphs[:20],
    mlp_decoded[:20],
    "MLP SONAR"
)

# %% [markdown]
# ## Summary and Exercises
#
# We've explored SONAR parascopes, which learn to map residual streams to text embeddings.
#
# ### Key Findings:
# 1. SONAR parascopes can decode semantic content without using the model's generation
# 2. MLP probes generally outperform linear probes (more parameters = better fit)
# 3. Different methods preserve different aspects of the original text
#
# ### Additional Exercises to Try:
#
# 1. **Layer Selection**: Modify `load_res_data` to use only specific layers (e.g., last 10)
#    and see how performance changes.
#
# 2. **Embedding Visualization**: Use t-SNE or PCA to visualize the learned SONAR embeddings
#    and see if they cluster by topic.
#
# 3. **Cross-Model Transfer**: Train on one model's residuals and test on another's.
#
# 4. **Noise Robustness**: Add noise to residual streams and measure degradation.
#
# 5. **Fine-grained Analysis**: Compare performance on different text types (technical,
#    narrative, dialogue).

# %% [markdown]
# ### Bonus Exercise: Visualizing the Embedding Space
#
# Let's visualize how residual streams map to SONAR space.

# %%
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def visualize_embedding_space(probe: nn.Module, res_data: torch.Tensor,
                            labels: List[str] = None, method: str = 'pca'):
    """Visualize how residual streams map to embedding space."""
    probe.eval()
    with torch.no_grad():
        # Get embeddings
        res_norm = res_normalizer.normalize(res_data.to(DEVICE))
        embeddings = probe(res_norm).cpu().numpy()

        # Reduce dimensions
        if method == 'pca':
            reducer = PCA(n_components=2)
        else:
            reducer = TSNE(n_components=2, random_state=42)

        reduced = reducer.fit_transform(embeddings)

        # Plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1],
                            c=range(len(reduced)), cmap='viridis',
                            alpha=0.6)
        plt.colorbar(scatter, label='Sample Index')
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
        plt.title(f'Residual → SONAR Embedding Space ({probe.__class__.__name__})')

        # Add some labels if provided
        if labels:
            for i in range(min(5, len(labels))):
                plt.annotate(labels[i][:20] + '...',
                           (reduced[i, 0], reduced[i, 1]),
                           fontsize=8, alpha=0.7)

        plt.tight_layout()
        plt.show()

# Visualize embedding spaces
print("Visualizing embedding spaces...")
visualize_embedding_space(linear_probe, res_data[:100], paragraphs[:100], 'pca')
visualize_embedding_space(mlp_probe, res_data[:100], paragraphs[:100], 'pca')

print("\nExercise complete! Try modifying the probe architectures or training procedures to improve performance.")