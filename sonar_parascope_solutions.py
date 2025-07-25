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
from torch.utils.data import DataLoader
from huggingface_hub import hf_hub_download
from datasets import load_dataset
import gc
import os

# For the model
from transformer_lens import HookedTransformer

# SONAR imports
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline

# For evaluation
from sentence_transformers import SentenceTransformer

# Disable gradients by default
torch.set_grad_enabled(False)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16
print(f"Using device: {DEVICE}, dtype: {DTYPE}")

# %% [markdown]
# ## Load the model

model = HookedTransformer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", device=DEVICE, dtype=DTYPE)

# %% [markdown]
# ## Exercise 1: Generating Training Data
#
# Since we want the probe to match "what the model might output", we need data of what the model's output might look like.
# Two-step process: (1) load dataset + generate prompts, (2) use prompts to generate model outputs.

# %% [markdown]
# ### Part 1: Generating Prompts

# %%
def format_prompt(prompt: list[str] | str, system_prompt: str = None) -> list[str]:
    def format_prompt_string(prompt: str) -> str:
        """Format prompt using the model's chat template."""
        messages = [
            {"role": "user", "content": prompt}
        ]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        return model.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    if isinstance(prompt, str):
        return [format_prompt_string(prompt)]
    elif isinstance(prompt, list):
        return [format_prompt_string(p) for p in prompt]

def format_question_string(text: str, max_chars: int = 32000) -> str:
    """Transform existing text into a generation prompt."""
    return f"""Content: {text[:max_chars]}

REQUEST: Write a prompt based on the above text, that is a single-paragraph, high-level description. Make the prompt in the format: "Write a (article/piece/entry), which includes (2-5 topics). The piece should be approximately (many n-paragraphs) long."

Only provide the prompt, do not write anything else."""

def format_question(text: list[str] | str, max_chars: int = 4000) -> list[str]:
    if isinstance(text, str):
        return [format_question_string(text, max_chars)]
    elif isinstance(text, list):
        return [format_question_string(t, max_chars) for t in text]
    else:
        raise ValueError(f"Invalid text type: {type(text)}")

# Load dataset and generate training data
print("Loading dataset...")
dataset = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
train_dataloader = DataLoader(dataset.take(4), batch_size=1)
# Note: for whatever reason with TransformerLens HookedTransformer,
# the generation does not work correctly with batching?
# TODO: make it work with batch > 1

# Step 1: Generate prompts from texts
prompts = []
for data in train_dataloader:  # Start with just 4 examples
    # Format the meta-prompt properly
    meta_prompt = format_question(data["text"])
    formatted_prompt = format_prompt(meta_prompt, system_prompt="You are a prompt-writing assistant. You are given a text and you need to write a prompt that will generate a response that is similar to the text.")

    # Tokenize and generate
    model.tokenizer.pad_token_id = model.tokenizer.eos_token_id
    prompt_tokens = model.to_tokens(
        formatted_prompt, prepend_bos=False, padding_side="left")
    generated_tokens = model.generate(
        prompt_tokens,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
    )

    # Extract just the generated part
    output_tokens = generated_tokens[:, len(prompt_tokens[0]):-1]
    generated_prompt = model.to_string(output_tokens)

    prompts.extend([t.strip() for t in generated_prompt])

print(f"Generated {len(prompts)} training examples")
print(f"Example prompts:")
[print(f"{i}: {[p]}") for i, p in enumerate(prompts)]

# %% [markdown]
# ### Part 2: Generate Model Outputs from Prompts

# %%
model_outputs = []
for i, prompt in enumerate(prompts):
    print(f"Generating output {i+1}/{len(prompts)}...")

    # Format and tokenize the prompt
    formatted_prompt = format_prompt(prompt)
    prompt_tokens = model.to_tokens(
        formatted_prompt, prepend_bos=False, padding_side="left")

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
print(f"Example outputs:")
[print(f"{i}: {[o]}") for i, o in enumerate(model_outputs)]

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

        if i > 100:
            # Lazy mode: only do 100 examples
            break
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
# - The actual paragraph text (we got this above)

# %% [markdown]
# ### Part 1: Loading Residual Streams
# So the transformer model has blocks of:
# [resid_pre] -> Attention -> [resid_mid] -> MLP -> [resid_post] == [resid_pre_n+1]
# I have only tested things so far with residual difference
# I.e: resid_mid = resid_pre + attn_results == resid_pre + resid_mid_diff == resid_pre + (resid_mid - resid_pre)
# I find it more consistent to calculate (option 1):
# * resid_mid_diff = resid_mid - resid_pre
# * resid_post_diff = resid_post - resid_mid
# alternatively, we could do (option 2):
# * resid_layer_diff = resid_post - resid_pre
# which should also work.
# My testing so far has used option 1, mostly so I can somewhat see where it is easier to extract the information.
# I suspect that the probes could work also using the basline activations [resid_mid, resid_post] or even just [resid_post], but I haven't tested it.

# %% [markdown]
# We make hooks for storing activations of the residual stream at the correct positions. That is, the final token of each "section" of tokens.
# For now, we just save [resid_pre, resid_mid, resid_post] at the end of each section.
# Later we can process it however we want.


def get_act_data(split_text, act_types=None, verbose=False):
    # choose which residual data to collect
    if act_types is None:
        act_types = ["hook_resid_pre", "hook_resid_mid", "hook_resid_post"]
    hook_names = [
        f"blocks.{i}.{resid_type}"
            for i in range(model.cfg.n_layers)
            for resid_type in act_types
    ]

    # get prompt vs output separately
    prompt = split_text[0]
    output = split_text[1:]

    # Tokenize the prompt and output correctly
    prompt_tokens =  model.to_tokens(prompt, prepend_bos=True)
    output_tokens = [model.to_tokens(o, prepend_bos=False) for o in output]
    if verbose:
        print(prompt_tokens.shape, [o.shape for o in output_tokens])
    all_tokens = torch.cat([prompt_tokens, torch.cat(output_tokens, dim=1)], dim=1)

    # Get the indices of the residual streams that we want to store
    # Ie: last token of each section, usually "\n\n"
    final_indices_rel = [
        prompt_tokens.shape[-1],
        *[ o.shape[-1] for o in output_tokens ]
    ]
    final_indices_abs = np.cumsum(final_indices_rel) - 1

    # check the tokens are actually the newline ones
    if verbose:
        print(model.to_str_tokens(all_tokens[:, final_indices_abs]))

    # Create hooks to store activations of only the correct residual streams
    act_data = {}
    def store_act(act, hook):
        act_data[hook.name] = act[..., final_indices_abs, :]
    hook_list = [(name, store_act) for name in hook_names]

    # Run model and store activations
    with model.hooks(fwd_hooks=hook_list):
        model.forward(all_tokens)

    # Print some info
    if verbose:
        for k, v in act_data.items():
            print(k, v.shape)
            break

    return act_data

act_data = get_act_data(split_sections[0]["split_text"], verbose=True)

# %%
# Now we actually save residual stream diff data, as describe as "option 1" above.

def get_resid_diff_data(split_text, act_types=None, verbose=False):
    act_data = get_act_data(split_text, act_types, verbose)

    # Calculate the difference between the residual streams
    act_data["resid_mid_diff"]  = [
        act_data[f"blocks.{i}.hook_resid_mid"] - act_data[f"blocks.{i}.hook_resid_pre"]
        for i in range(model.cfg.n_layers)
    ]
    act_data["resid_post_diff"] = [
        act_data[f"blocks.{i}.hook_resid_post"] - act_data[f"blocks.{i}.hook_resid_mid"]
        for i in range(model.cfg.n_layers)
    ]

    # Concatenate the data into a single tensor
    # I previously did this as [batch==1, layers, tokens, d_model]
    # where layers is ordered as [resid_pre_0] + \
    # [mid_diff_0, post_diff_0, mid_diff_1, post_diff_1, ...]
    # So for consistency, we store it this way.
    full_act_data = [act_data["blocks.0.hook_resid_pre"]]
    for i in range(model.cfg.n_layers):
        full_act_data.append(act_data["resid_mid_diff"][i])
        full_act_data.append(act_data["resid_post_diff"][i])

    return torch.cat(full_act_data, dim=0).unsqueeze(0)

resid_diff_data = get_resid_diff_data(split_sections[0]["split_text"], verbose=True)
resid_diff_data.shape

# note: I kinda wish I stored things as raw residuals instead of diffs, since it's easier to convert to diffs later.

# We are now finished with the model so we can remove it from memory.
del model
torch.cuda.empty_cache()

# %%
# Now we load the pre-computed residual stream diffs.


# Download and load a specific file (there are total 100 files, 000-099)
# Note these are slightly older files, so these match up with a different dataset.

split_sections = load_dataset("nickypro/llama-3b-split", split="train")
print(split_sections[0])

file_path = hf_hub_download(
    repo_id="nickypro/llama-3b-residuals",
    filename="res_data_000.pt",
    repo_type="dataset"
)

# Load the tensor
tensor_data = torch.load(file_path)
print(f"Loaded tensor shape: {tensor_data[0].shape}")

# %%
# ## Creating the embeddings.
# Now we use SONAR to create embeddings for each paragraph.
# We use the `sonar.inference_pipelines.text.TextToEmbeddingModelPipeline` class to create the embeddings.
# We do this for all the paragraphs in the dataset.

# Note again we use a slightly older dataset because I haven't uploaded the new ones to huggingface yet.
split_sections = load_dataset("nickypro/llama-3b-split", split="train")
print(len(split_sections[0]['split']))

text2vec = TextToEmbeddingModelPipeline(
    encoder="text_sonar_basic_encoder",
    tokenizer="text_sonar_basic_encoder",
    device=DEVICE,
    dtype=DTYPE,
)

vec2text = EmbeddingToTextModelPipeline(
    decoder="text_sonar_basic_decoder",
    tokenizer="text_sonar_basic_encoder",
    device=DEVICE,
    dtype=DTYPE,
)

# %% We try to get the embeddings, and also compare some examples of how well it decodes.

embeds = []

for i, example in enumerate(tqdm(split_sections)):
    _id = example["id"]
    split_texts = example["split"]
    embeddings = text2vec.predict(split_texts, source_lang="eng_Latn")
    embeds.append(embeddings)
    print(embeddings.shape)
    decoded_texts = vec2text.predict(embeddings, target_lang="eng_Latn")
    for j, t in enumerate(split_texts):
        print("original:  ", [t])
        print("predicted: ", [decoded_texts[j]])
    break

# %% alternatively, we can yet again use the pre-computed embeddings.

embeds = torch.load(hf_hub_download(
    repo_id="nickypro/llama-3b-embeds",
    filename="embeds_000.pt",
    repo_type="dataset"
))
print(embeds[0].shape)

for i, embed in enumerate(embeds):
    decoded_texts = vec2text.predict(embed.to(DTYPE), target_lang="eng_Latn")
    print(f"original:  {[split_sections[i]['split'][1]]}")
    print(f"predicted: {[decoded_texts[0]]}")
    break

# %% [markdown]
# Now we have shown how to get the data, we can remove the text2vec pipeline from memory. and instead use the pre-computed embeddings.
# %%
del text2vec
torch.cuda.empty_cache()

# %%
# Now we just load all data.
# Note that this data only saves:
# - residuals[:-1]
# - embeds[1:]
# since we only use each residual to predict the next embedding.
# Thus they should match up already.

def load_all_data(index: int = 0):
    split_sections = load_dataset("nickypro/llama-3b-split", split="train")

    res_data_file_path = hf_hub_download(
        repo_id="nickypro/llama-3b-residuals",
        filename=f"res_data_{index:03d}.pt",
        repo_type="dataset"
    )
    res_data = torch.load(res_data_file_path, map_location='cpu')

    embeds_file_path = hf_hub_download(
        repo_id="nickypro/llama-3b-embeds",
        filename=f"embeds_{index:03d}.pt",
        repo_type="dataset"
    )
    embeds = torch.load(embeds_file_path, map_location='cpu')

    assert len(res_data) == len(embeds)
    dataset = []
    res_reshape = "1 layer section dim -> section layer dim"
    for i, (res, embed) in enumerate(zip(res_data, embeds)):
        _id = i + 1000 * index
        dataset.append({
            "id": _id,
            "res_data": einops.rearrange(res, res_reshape),
            "embeds": embed,
            "split_text": split_sections[_id]["split"],
        })
    return dataset

dataset = load_all_data()

print(len(dataset))
print(dataset[0]["id"])
print(dataset[0]["res_data"].shape)
print(dataset[0]["embeds"].shape)
[print([p]) for p in dataset[0]["split_text"]]

# %% [markdown]
# Ok now we have all the data we need. So we can start training, right?

# %%

# %%# %% [markdown]
# ## Exercise 3: Normalization and Preprocessing
#
# One issue with residual streams, is that the magnitudes of the activations can vary a lot between layers, often by orders of magnitude.
# This can cause issues for training, so we need to normalize the data.
# We use Welford's algorithm to compute running statistics.
# We compute the mean and variance of the residual streams and embeddings, and then normalize the data to have mean 0 and variance 1.
# We then store the mean and variance, so we can restore the data later.
#
# In essense, we do the most naive method of normalization, which is to look at each dimension independently and normalize it to have mean 0 and variance 1.
# There may be better ways do do this, I have not spent much time optimizing this.
# While we do need to normalize the residuals, I am not sure if we need to do it for the embeddings, but I do it anyway.

# %%
@dataclass
class WelfordStats:
    """Track running mean and variance using Welford's algorithm."""
    mean: torch.Tensor
    m2: torch.Tensor
    count: int

    def __init__(self, mean: torch.Tensor = None, m2: torch.Tensor = None, count: int = 0):
        if mean is not None and m2 is not None:
            self.mean = mean
            self.m2 = m2
            self.count = count
        else:
            self.mean = None
            self.m2 = None
            self.count = 0

    def update(self, new_data: torch.Tensor):
        """Update statistics with new batch of data (batched version, true Welford)."""
        # new_data: (batch, d)
        if self.mean is None or self.m2 is None:
            self.mean = torch.zeros_like(new_data[0])
            self.m2 = torch.zeros_like(new_data[0])
            self.count = 0
        for x in new_data:
            self.count += 1
            delta = x - self.mean
            self.mean += delta / self.count
            delta2 = x - self.mean
            self.m2 += delta * delta2

    @property
    def sample_variance(self):
        # Unbiased sample variance
        return self.m2 / (self.count - 1) if self.count > 1 else torch.zeros_like(self.m2)

    @property
    def population_variance(self):
        # Population variance
        return self.m2 / self.count if self.count > 0 else torch.zeros_like(self.m2)

    @property
    def std(self):
        # Use sample variance by default
        return torch.sqrt(self.sample_variance + 1e-6)

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
res_stats = WelfordStats()
embed_stats = WelfordStats()

# Update with data
for i, example in enumerate(tqdm(dataset)):
    res_stats.update(example["res_data"])
    embed_stats.update(example["embeds"])

# Create normalizers
res_normalizer = Normalizer(res_stats.mean, res_stats.std, device='cpu')
embed_normalizer = Normalizer(embed_stats.mean, embed_stats.std, device='cpu')

# Test normalization
def test_normalization(dataset):
    normalized_res    = res_normalizer.normalize(dataset[0]["res_data"])
    normalized_embeds = embed_normalizer.normalize(dataset[0]["embeds"])
    print(f"Normalized residual mean: {normalized_res.mean():.4f}, std: {normalized_res.std():.4f}")
    print(f"Normalized embeds mean: {normalized_embeds.mean():.4f}, std: {normalized_embeds.std():.4f}")

test_normalization(dataset)

# %% [markdown]
# ## Exercise 4: Define Probe Models
#
# We'll implement a simple Linear probes to map from residual streams to SONAR embeddings.
# We could take all of the layers [0..57] but I found diminishing returns after 24 layers.
# I also have tried MLPs, but their performance was basically identical to the linear probe.

# %%
class LinearProbe(nn.Module):
    """Simple linear mapping from residual stream to SONAR embedding."""
    def __init__(self, d_res: int = 3072, d_sonar: int = 1024, n_layers_to_use: int = 24):
        super().__init__()
        self.n_layers_to_use = n_layers_to_use
        d_in = d_res * self.n_layers_to_use
        self.linear = nn.Linear(d_in, d_sonar)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # use the last n_layers_to_use layers of residual diffs
        x = x[..., -self.n_layers_to_use:, :].flatten(start_dim=-2)
        return self.linear(x)

# Create probe models
d_res = dataset[0]["res_data"].shape[-1]
linear_probe = LinearProbe(d_res).to(DEVICE, DTYPE)

print(f"Linear probe parameters: {sum(p.numel() for p in linear_probe.parameters()):,}")

# %% [markdown]
# ## Exercise 5: Training Loop
#
# We now train the probe to map from [residual stream] to [SONAR embedding].
# For efficiency, we currently load data from 10,000 texts (index=0...9), but this could be extended to 100,000 (index=0...99).
# We use index 99 as a relatively independent validation set, and validate every epoch.
#

# %%

class ProbeTrainer:
    def __init__(
        self,
        probe: nn.Module,
        lr: float = 1e-5,
        weight_decay: float = 1e-6,
        lr_decay: float = 0.8,
        batch_size: int = 1024,
        dtype=DTYPE,
        device=DEVICE,
        checkpoint_dir: str = "./checkpoints",
    ):
        self.probe = probe
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_decay = lr_decay
        self.batch_size = batch_size
        self.dtype = dtype
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        # Training components
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.probe.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=1,
            gamma=self.lr_decay
        )

        # Ensure checkpoint directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    @staticmethod
    def preprocess_dataset(dataset: list[dict]):
        """Convert dataset to tensors and normalize."""
        dataset_dict = {
            "texts": [example["split_text"][1:] for example in dataset],
            "res_data": res_normalizer.normalize(torch.cat([example["res_data"] for example in dataset])),
            "embeds": embed_normalizer.normalize(torch.cat([example["embeds"] for example in dataset])),
        }
        return dataset_dict

    def get_dataloader(self, res_data, embeds, shuffle=True):
        """Create DataLoader with proper memory management."""
        dataset = torch.utils.data.TensorDataset(res_data, embeds)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        return loader

    def train_epoch(self, epoch: int, train_indices: List[int]) -> float:
        """Train for one epoch with improved memory management."""
        self.probe.train()
        epoch_train_loss = 0
        n_train_batches = 0

        pbar = tqdm(train_indices, desc=f"Train Epoch {epoch+1}")
        for data_idx in pbar:
            try:
                # Load data for this file
                dataset = load_all_data(data_idx)
                dataset_dict = self.preprocess_dataset(dataset)
                res_data = dataset_dict["res_data"]
                embeds = dataset_dict["embeds"].to(self.dtype)

                # Create dataloader
                loader = self.get_dataloader(res_data, embeds, shuffle=True)

                # Training loop for this file
                for batch_x, batch_y in loader:
                    batch_x = batch_x.to(self.device, non_blocking=True)
                    batch_y = batch_y.to(self.device, non_blocking=True)

                    # Forward pass
                    self.optimizer.zero_grad()
                    pred = self.probe(batch_x)
                    loss = self.criterion(pred, batch_y)

                    # Backward pass
                    loss.backward()
                    self.optimizer.step()

                    epoch_train_loss += loss.item()
                    n_train_batches += 1

                    # Update progress bar
                    current_avg_loss = epoch_train_loss / n_train_batches
                    pbar.set_postfix({
                        "Loss": f"{current_avg_loss:.4f}",
                        "LR": f"{self.scheduler.get_last_lr()[0]:.2e}"
                    })

                # Clean up memory after each file
                del dataset, dataset_dict, res_data, embeds, loader
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error processing file {data_idx}: {e}")
                continue

        return epoch_train_loss / max(n_train_batches, 1)

    @torch.no_grad()
    def validate(self, val_indices: List[int]) -> float:
        """Validate the model with improved memory management."""
        self.probe.eval()
        epoch_val_loss = 0
        n_val_batches = 0

        with torch.no_grad():
            for data_idx in tqdm(val_indices, desc="Validation"):
                try:
                    # Load validation data
                    dataset = load_all_data(data_idx)
                    dataset_dict = self.preprocess_dataset(dataset)
                    res_data = dataset_dict["res_data"]
                    embeds = dataset_dict["embeds"].to(self.dtype)

                    # Create dataloader
                    loader = self.get_dataloader(res_data, embeds, shuffle=False)

                    # Validation loop for this file
                    for batch_x, batch_y in loader:
                        batch_x = batch_x.to(self.device, non_blocking=True)
                        batch_y = batch_y.to(self.device, non_blocking=True)

                        pred = self.probe(batch_x)
                        loss = self.criterion(pred, batch_y)
                        epoch_val_loss += loss.item()
                        n_val_batches += 1

                    # Clean up memory after each file
                    del dataset, dataset_dict, res_data, embeds, loader
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception as e:
                    print(f"Error processing validation file {data_idx}: {e}")
                    continue

        return epoch_val_loss / max(n_val_batches, 1)

    def train(
        self,
        num_epochs: int = 1,
        train_indices: List[int] = list(range(0, 99)),
        val_indices: List[int] = [99],
        save_checkpoints: bool = True,
        validate_every: int = 1,
    ) -> Dict[str, List[float]]:
        """
        Main training loop with improved features.

        Args:
            num_epochs: Number of epochs to train
            train_indices: List of data file indices for training
            val_indices: List of data file indices for validation
            save_checkpoints: Whether to save model checkpoints
            validate_every: Validate every N epochs

        Returns:
            Dictionary containing training and validation losses
        """
        torch.set_grad_enabled(True)

        train_losses = []
        val_losses = []

        print(f"Starting training for {num_epochs} epochs")
        print(f"Training files: {len(train_indices)}, Validation files: {len(val_indices)}")
        print(f"Initial LR: {self.lr}, LR Decay: {self.lr_decay}")

        try:
            for epoch in range(num_epochs):
                print(f"\nEpoch {epoch+1}/{num_epochs}")

                # Training
                train_loss = self.train_epoch(epoch, train_indices)
                train_losses.append(train_loss)

                # Validation
                if epoch % validate_every == 0 or epoch == num_epochs - 1:
                    val_loss = self.validate(val_indices)
                    val_losses.append(val_loss)

                    print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

                    # Save checkpoint
                    if save_checkpoints:
                        checkpoint_path = os.path.join(
                            self.checkpoint_dir,
                            f"probe_epoch_{epoch+1}.pkl"
                        )
                        self.save_checkpoint(checkpoint_path, epoch, train_loss, val_loss)
                else:
                    print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}")

                # Step the learning rate scheduler
                self.scheduler.step()

        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        except Exception as e:
            print(f"\nTraining error: {e}")
            raise
        finally:
            torch.set_grad_enabled(False)

        return {
            "train_losses": train_losses,
            "val_losses": val_losses
        }

    def save_checkpoint(self, checkpoint_path: str, epoch: int, train_loss: float, val_loss: float):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.probe.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        }, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path)
        self.probe.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint

linear_probe = LinearProbe(d_res).to(DEVICE, DTYPE)

# Use the improved ProbeTrainer
trainer = ProbeTrainer(
    probe=linear_probe,
    lr=5e-5,
    lr_decay=0.8,
    batch_size=1024,
    checkpoint_dir="./probe_checkpoints"
)

# Train with improved features
losses = trainer.train(
    num_epochs=10,
    train_indices=list(range(0, 99)),  # Reduced for demo
    val_indices=[99],
    save_checkpoints=True,
    validate_every=1
)

# %% [markdown]
# Test performance of the probe

try:
    get_name = lambda x: [k for k,v in globals().items() if v is x][0]
    print(f"{get_name(vec2text)} already loaded")
except:
    vec2text = EmbeddingToTextModelPipeline(
        decoder="text_sonar_basic_decoder",
        tokenizer="text_sonar_basic_encoder",
        device=DEVICE,
        dtype=DTYPE,
    )

# %%

dataset_dict = ProbeTrainer.preprocess_dataset(load_all_data(99))

for i, (text, res, emb) in list(enumerate(zip(dataset_dict["texts"], dataset_dict["res_data"], dataset_dict["embeds"])))[:100:5]:
    output = linear_probe(res.to(DEVICE)).to('cpu')
    emb    = emb.to('cpu')
    print(f"{i}: Mean Squared Error: {torch.nn.functional.mse_loss(emb, output)}")
    print(f"{i}: Cosine similarity: {torch.nn.functional.cosine_similarity(emb, output, dim=0)}")
    output = embed_normalizer.restore(output).to(DEVICE, DTYPE)
    emb    = embed_normalizer.restore(emb).to(DEVICE, DTYPE)
    predictions = vec2text.predict([emb, output], target_lang="eng_Latn")
    print(f"{i}: Original decoded: {predictions[0]}")
    print(f"{i}: Predicted decoded: {predictions[1]}")
    print()

# %%
# You should see that some predictions are pretty similar to the original text.
# And some of them are completelly broken tbh.
# You can get better results if you increase train_indices to be range(0, 99), but this will take a while.

