# %%
import transformers
from transformer_lens import HookedTransformer
import torch
torch.set_grad_enabled(False)

model = HookedTransformer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
# model = HookedTransformer.from_pretrained("google/gemma-2-9b-it")

# %%
# First run: capture attention activations
print("Available hook points:")
for name in list(model.hook_dict.keys())[:32]:  # Show first 10 hook points
    print(name)

# %%
# Example string to work with
prompt = \
"""Question:

Please write two paragraphs, each of length 2 sentences. The first should describe monster trucks, and the second should describe paracetamol. Do not say explicitly name the topic, and do not say anything else.

Answer:

Gigantic vehicles with massive tires and powerful engines dominate the racing scene, crushing cars and other obstacles with ease. Their massive size and strength make them a thrilling sight to behold for fans of high-speed competition.\n\n"""

example_string = prompt #+ par1

chosen_hook = 'hook_resid_pre'
chosen_hook_2 = 'hook_resid_mid'
global_data = {}

# Tokenize the example string
tokens = model.to_tokens(example_string)
# print(f"Tokens: {tokens}")
print(f"Token strings: {model.to_str_tokens(example_string)}")

# Generate context based on the string
orig_generated_tokens = model.generate(tokens, max_new_tokens=50, do_sample=True, temperature=0.3)

# Get the full string (example + generated text)
# print(orig_generated_tokens)

full_string = model.to_string(orig_generated_tokens)
print(f"Full generated string: {model.to_str_tokens(orig_generated_tokens)}")
print(full_string)


# Run with hooks to get activations of attention output
def store_attn_hook(attn_act, hook):
    # Store attention output activations
    global_data[hook.name] = attn_act.clone()
    return attn_act


# Create hooks for all layers
Ls = range(model.cfg.n_layers)
hook_list = [
    *[(f'blocks.{i}.{chosen_hook}', store_attn_hook) for i in Ls],
    *[(f'blocks.{i}.{chosen_hook_2}', store_attn_hook) for i in Ls]
]

# with model.hooks(fwd_hooks=[('blocks.0.ln1.hook_normalized', lambda resid_post, hook: print("SHAPE", resid_post.shape, hook.name, resid_post.flatten()[-6:]))]):
#     print(generated_tokens.shape)
#     logits = model(generated_tokens, return_type="logits")

with model.hooks(fwd_hooks=hook_list):
    logits = model(tokens, return_type="logits")


print(f"Stored activations for layers: {list(global_data.keys())}")

# %%
# Run with hooks again to modify activations of attention output
new_str = "\n"
new_tokens = model.to_tokens([new_str for _ in range(10)])
has_been_modified = set()

def modify_attn_hook(attn_act, hook):
    # Modify attention output activations (example: add small noise)
    if attn_act.shape[1] <= 1:
        return attn_act
    source_act = global_data[hook.name]
    attn_act[:, -1, :] = source_act[:, -1, :] * 1.5
    if hook.name not in has_been_modified:
        has_been_modified.add(hook.name)
        assert attn_act.shape[1] == 2
    return attn_act

# Second run: modify attention activations for all layers
modify_hook_list = [
    *[(f'blocks.{i}.{chosen_hook}', modify_attn_hook) for i in Ls],
    *[(f'blocks.{i}.{chosen_hook_2}', modify_attn_hook) for i in Ls]
]

with model.hooks(fwd_hooks=modify_hook_list):
    new_generated_tokens = model.generate(new_tokens, max_new_tokens=15, do_sample=True, temperature=0.3)

print(f"New generated tokens:")
for new_generated_token in new_generated_tokens:
    print(model.to_str_tokens(new_generated_token))
# %%
