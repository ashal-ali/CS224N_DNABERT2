!pip install transformers accelerate sentencepiece torch torchvision torchaudio
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, LlamaForCausalLM, LlamaTokenizer

!git clone https://github.com/ashal-ali/CS224N_DNABERT2.git

# Load Llama-2-7b-hf model & tokenizer
llama_model_name = "NousResearch/Llama-2-7b-chat-hf"
llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model_name)
llama_model = LlamaForCausalLM.from_pretrained(llama_model_name, torch_dtype=torch.float16, device_map="auto")

print(f"Llama-2-7b Model Loaded: {llama_model_name}")

%cd /content/CS224N_DNABERT2/CS224N_DNABERT2
!ls

from transformers import AutoTokenizer, AutoModel

# Define local directory path
dnabert_model_name = "/content/CS224N_DNABERT2/CS224N_DNABERT2"

# Load tokenizer and model from the correct path
dnabert_tokenizer = AutoTokenizer.from_pretrained(dnabert_model_name, local_files_only=True)
dnabert_model = AutoModel.from_pretrained(dnabert_model_name, trust_remote_code=True, local_files_only=True)

print(f"✅ Patched DNABERT-2 Model Loaded Successfully from {dnabert_model_name}")

import torch

# Move model to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
dnabert_model.to(device)

# Example DNA sequence
test_sequence = "TGCATG"
inputs = dnabert_tokenizer(test_sequence, return_tensors="pt")

# Move inputs to the same device as model
inputs = {key: value.to(device) for key, value in inputs.items()}

# Run model inference
outputs = dnabert_model(**inputs)

# Print output shape
print("Output shape:", outputs.last_hidden_state.shape)


Current error: OutOfResources: out of resource: shared memory, Required: 82944, Hardware limit: 65536. Reducing block sizes or `num_stages` may help.