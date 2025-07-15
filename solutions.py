# %% [markdown]
# Simple impementation of Continuation ParaScope.
#
# %%
from transformer_lens import HookedTransformer
import torch
torch.set_grad_enabled(False)

model = HookedTransformer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

# print("Available hook points:")
# for name in list(model.hook_dict.keys())[:28]:  # Show first few hook points
#     print(name)

# %% [markdown]
# ### Prompting and relevant scaffolding
# Some functions to help with prompting

# %%
def format_prompt(prompt: str) -> torch.Tensor:
    """
    Create tokenized chat using the model's tokenizer with chat template.
    """
    messages = [
        {"role": "user", "content": prompt}
    ]

    # Apply chat template and tokenize
    return model.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

def two_topics_prompt(topic1: str, topic2: str) -> torch.Tensor:
    prompt = f"""
Please write two paragraphs, each of length 2 sentences. The first should describe {topic1}, and the second should describe {topic2}. Do not say explicitly name the topic, and do not say anything else.
"""
    return format_prompt(prompt)

def extract_second_paragraph(text: str) -> str:
    return text.split("\n\n")[1]

# %% [markdown]
# ### Baseline Generation
# Try running the model with just the prompt to get a baseline.
# %%

prompt = two_topics_prompt("monster trucks", "paracetamol")
prompt_tokens = model.to_tokens(prompt)[:, 1:] # remove double <bos>

# Generate context based on the string
baseline_generated_tokens = model.generate(prompt_tokens, max_new_tokens=100, do_sample=True, temperature=0.3)
output_tokens = baseline_generated_tokens[:, len(prompt_tokens[0]):]
output_str_tokens = model.to_str_tokens(output_tokens)
print(model.to_str_tokens(prompt_tokens))
print(output_str_tokens)

# get first paragraph
# Find the index of the first occurrence of '\n\n' in the output tokens
index_of_first_newline = None
for i, token in enumerate(output_str_tokens):
    if '\n\n' in token:
        index_of_first_newline = i
        break

assert index_of_first_newline is not None
print(f"Index of first newline: {index_of_first_newline} ({[output_str_tokens[index_of_first_newline]]})")

par1_tokens = output_tokens[:, :index_of_first_newline+1]
prompt_with_par1 = torch.cat((prompt_tokens, par1_tokens), dim=-1)

# %% [markdown]
# Define the hooks that we'll use to capture the activations
#
# We'll use the `store_hook` method to store the activations, and the `modify_hook` method to modify the activations.
#
# The `modify_hook` method is given the activation and the hook, and should return the modified activation only if the hook has not been seen before. (this prevents the scenario where you keep getting the same output over and over)

class ActStore:
    def __init__(self, model: HookedTransformer):
        self.act_store = {}
        self.act_seen = set()

        chosen_hooks = ['hook_resid_pre', 'hook_resid_mid']
        Ls = range(model.cfg.n_layers) # all layers, you can test with fewer layers
        self.chosen_hook_list = \
            [f'blocks.{i}.{h}' for h in chosen_hooks for i in Ls]
        self.modify_scaling_factor = 1.0
        self.modify_hook_token_idx = -1

    def store_hook(self, act, hook):
        self.act_store[hook.name] = act.clone()
        return act

    def modify_hook(self, act, hook):
        # should we modify the act, or have we already done so?
        if hook.name in self.act_seen:
            return act
        self.act_seen.add(hook.name)

        # modify the act
        source_act = self.act_store[hook.name]
        idx = self.modify_hook_token_idx
        scale = self.modify_scaling_factor
        act[:, idx, :] = source_act[:, -1, :] * scale
        return act

    def modify_hook_list(self):
        "return a list of hooks to modify, and reset the act_seen set to empty"
        self.act_seen = set()
        return [(hook, self.modify_hook) for hook in self.chosen_hook_list]

    def store_hook_list(self):
        "return a list of hooks to store"
        return [(hook, self.store_hook) for hook in self.chosen_hook_list]

act_store = ActStore(model)

# %% [markdown]
# ## Generate with transferred activations.
# This technique demonstrates "activation patching" or "activation transfer" - a method for transferring learned context from one prompt to another by manipulating the model's internal representations.
#
# The process works as follows:
# 1. Run the model on a "source" prompt that contains rich context (e.g., a paragraph with specific style/content)
# 2. Store the intermediate activations (residual stream values) from key layers during this forward pass
# 3. Run the model on a "target" prompt that lacks context (e.g., just "\n\n")
# 4. During the target generation, replace the activations at the same positions with the stored activations from the source
# 5. This allows the model to generate text as if it had the original context, even though the target prompt is minimal
#
# This technique reveals how much contextual information is encoded in the model's intermediate representations
# and can be used to study how different types of context (style, topic, format) are stored and utilized.

# Get the full string (example + generated text)

def transfer_activations(tokens, num_copies=10):
    """
    function that runs the model once with the original prompt, saves the activations, then runs the model again with the new prompt, modifying the activations to match the original.
    """

    # Save original activations
    with model.hooks(fwd_hooks=act_store.store_hook_list()):
        logits = model(tokens, return_type="logits")

    # create prompt
    new_str = "\n\n"
    new_tokens = model.to_tokens([new_str for _ in range(num_copies)])

    # Run with hooks again to modify activations of attention output
    with model.hooks(fwd_hooks=act_store.modify_hook_list()):
        new_generated_tokens = model.generate(new_tokens, max_new_tokens=15, do_sample=True, temperature=0.3)

    return new_generated_tokens

# %% [markdown]
# ## Demonstrating Activation Transfer
#
# Now we'll test our activation transfer function on different prompts to see how well it preserves
# and transfers contextual information. We'll run the function on:
# 1. The original single paragraph prompt to see how it continues the established style/content
# 2. The two-paragraph prompt to observe how it handles more complex context transfer
#
# Each test will generate multiple continuations to show the consistency and variability of the transfer.

# try for paragraph 1
par1_continued_tokens = transfer_activations(prompt_tokens)
for tok in par1_continued_tokens:
    print(model.to_str_tokens(tok))

# try for paragraph 2
par2_continued_tokens = transfer_activations(prompt_with_par1)
for tok in par2_continued_tokens:
    print(model.to_str_tokens(tok))

# %% [markdown]
# ## LoRA Fine-tuning
#
# Now we'll try to fine-tune the model to be more consistent in focusing on the transferred activations.
#


# %% Define the LoRA block and Lora hooks needed

import torch
import torch.nn as nn
import einops
from typing import Callable
from jaxtyping import Float
from torch import Tensor
from transformer_lens.hook_points import HookPoint
from transformer_lens import HookedTransformerConfig

DEBUG = False  # Add debug flag

class Lora(nn.Module):
    """
    Module that implements the basic LoRA block.
    - Input: tensor of shape (..., [inst], d_in) and returns a tensor of shape (..., inst, d_out).
    - Calculated intermediate activations of shape (..., inst, rank)
    - Output: tensor of shape (..., inst, d_out)
    """
    A: nn.Parameter
    B: nn.Parameter

    def __init__(self,
            n_inst: int | None = None,
            d_in: int = 768,
            d_out: int = 768,
            rank: int = 4,
            lora_alpha: float = 32,
            add_inst_dim: bool = False,
            dtype: torch.dtype | None = None):
        """
        Initialize the weights of the LoRA block.
        - The A block should be initialized with kaiming uniform with a=sqrt(5)
        - The B block should be initialized with zeros.
        """
        super().__init__()
        self.rank = rank
        self.n_inst = n_inst
        self.lora_alpha = lora_alpha
        self.add_inst_dim = add_inst_dim
        self.A = nn.Parameter(torch.empty(n_inst, d_in, rank, dtype=dtype))
        self.B = nn.Parameter(torch.zeros(n_inst, rank, d_out, dtype=dtype))
        self.dtype = dtype

        nn.init.kaiming_uniform_(self.A, a = 5**0.5)

    def forward(self, x: Float[Tensor, "... [inst] d_in"]) -> Float[Tensor, "... inst d_out"]:
        """
        Computes the forward pass of the LoRA block.
        - if add_inst_dim is True, input is of shape "... d_in". Need to repeat across instance dimension.
        - if add_inst_dim is False, input is of shape "... inst d_in". Each instance dimension already exists.
        """
        # orig_dtype = x.dtype
        if x.dtype != self.dtype:
            x = x.to(self.dtype)

        if DEBUG:
            print(f"LoRAattn input: {x.shape=} {x.dtype=}")
        if self.add_inst_dim:
            x = einops.repeat(x, "... d_in -> ... inst d_in", inst = self.n_inst)
        assert x.shape[-2] == self.n_inst
        out = einops.einsum(x, self.A, self.B, "... inst d_in, inst d_in rank, inst rank d_out -> ... inst d_out")

        # out = out.to(orig_dtype)

        if DEBUG:
            print(f"LoRAattn output: {out.shape=} {out.dtype=}")
        return out * self.lora_alpha / self.rank

test_lora = Lora(n_inst=1, d_in=10, d_out=20, rank=2)
assert sum(p.numel() for p in test_lora.parameters()) == (10*2 + 2*20)
assert torch.allclose(test_lora.B, torch.zeros_like(test_lora.B))
assert not torch.allclose(test_lora.A, torch.zeros_like(test_lora.A))

class LoraAttentionHooks(nn.ModuleDict):
    """
    Defines the LoRA hooks needed for the Attention Layers of the transformer.
    """
    def __init__(self,
                 cfg: HookedTransformerConfig,
                 lora_alpha: float = 32,
                 rank: int = 4,
                 dtype: torch.dtype = None):
        super().__init__()
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.dtype = dtype

        # Note: In newer models, the keys and values are shared across multiple attention heads
        # (n_key_value_heads <= n_qo_heads = n_heads), which reduces parameters while maintaining
        # performance. This is why we have separate n_qo_heads and n_kv_heads variables.
        n_qo_heads = cfg.n_heads
        n_kv_heads = cfg.n_key_value_heads if cfg.n_key_value_heads is not None else n_qo_heads
        d_model, d_head, d_mlp = cfg.d_model, cfg.d_head, cfg.d_mlp

        self['hook_q'] = Lora(n_qo_heads, d_model, d_head, rank=rank, lora_alpha=lora_alpha, add_inst_dim=True, dtype=dtype)
        self['hook_k'] = Lora(n_kv_heads, d_model, d_head, rank=rank, lora_alpha=lora_alpha, add_inst_dim=True, dtype=dtype)
        self['hook_v'] = Lora(n_kv_heads, d_model, d_head, rank=rank, lora_alpha=lora_alpha, add_inst_dim=True, dtype=dtype)
        self['hook_o'] = Lora(n_qo_heads, d_head, d_model, rank=rank, lora_alpha=lora_alpha, add_inst_dim=False, dtype=dtype)

    def store_hook_normalized(self, normalized: Float[Tensor, "batch pos d_model"], hook: HookPoint) -> None:
        self.cache_qkv = normalized

    def store_hook_z(self, z: Float[Tensor, "batch pos n_heads d_head"], hook: HookPoint) -> None:
        self.cache_z = z

    def adjust_hook_qkv(self, qkv: Float[Tensor, "batch pos n_heads d_head"], hook: HookPoint) -> Float[Tensor, "batch pos n_heads d_head"]:
        name = hook.name.split('.')[-1]
        return qkv + self[name](self.cache_qkv)

    def adjust_hook_out(self, attn_out: Float[Tensor, "batch pos n_heads d_head"], hook: HookPoint) -> Float[Tensor, "batch pos n_heads d_head"]:
        """
        NOTE: Due to
        """
        lora_result = self['hook_o'](self.cache_z)
        orig_dtype = lora_result.dtype
        lora_result = lora_result.to(self.dtype)
        lora_attn_out = einops.einsum(lora_result, "... n_heads d_model -> ... d_model")
        lora_attn_out = lora_attn_out.to(orig_dtype)
        return attn_out + lora_attn_out

    def list_fwd_hooks(self, layer_idx: int) -> list[tuple[str, Callable]]:
        """
        Returns a list of hook_point names and functions to call for the forward pass of
        the model using LoRA.
        """
        fwd_hooks = []
        hook_in_name = f'blocks.{layer_idx}.ln1.hook_normalized'
        fwd_hooks.append((hook_in_name, self.store_hook_normalized))

        for key in ["hook_q", "hook_k", "hook_v"]:
            hook_out_name = f'blocks.{layer_idx}.attn.{key}'
            fwd_hooks.append((hook_out_name, self.adjust_hook_qkv))

        hook_in_name = f'blocks.{layer_idx}.attn.hook_z'
        hook_out_name = f'blocks.{layer_idx}.hook_attn_out'
        fwd_hooks.append((hook_in_name, self.store_hook_z))
        fwd_hooks.append((hook_out_name, self.adjust_hook_out))
        return fwd_hooks

class LoraAllLayers(nn.Module):
    def __init__(self, cfg: HookedTransformerConfig, lora_alpha: float = 32, rank: int = 4, dtype: torch.dtype = None):
        super().__init__()
        self.lora_hooks = nn.ModuleList([LoraAttentionHooks(cfg, lora_alpha, rank, dtype) for _ in range(cfg.n_layers)])

    def list_fwd_hooks(self) -> list[tuple[str, Callable]]:
        """
        Returns a list of hook_point names and functions to call for the forward pass of
        all layers in the model using LoRA.
        """
        fwd_hooks = []
        for layer_idx in range(len(self.lora_hooks)):
            fwd_hooks.extend(self.lora_hooks[layer_idx].list_fwd_hooks(layer_idx))
        return fwd_hooks

# %%
lora = LoraAllLayers(model.cfg, lora_alpha=32, rank=4)

with model.hooks(fwd_hooks=lora.list_fwd_hooks()):
    logits = model(prompt_tokens, return_type="logits")

# %%
