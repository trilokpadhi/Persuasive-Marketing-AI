import os
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM, AutoConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import hf_hub_download, snapshot_download
import torch

MODEL_NAME = f"meta-llama/Llama-2-70b-hf"
WEIGHTS_DIR = f"{os.getcwd()}/llama-weights-70b"


# Download model

if not os.path.exists(WEIGHTS_DIR):
    os.system(f"mkdir {WEIGHTS_DIR}")

#checkpoint_location = snapshot_download(MODEL_NAME, local_dir=WEIGHTS_DIR, ignore_patterns=["*.safetensors", "model.safetensors.index.json"], token="my_token_goes_here") # run this if you haven't downloaded the 70b model
checkpoint_location = WEIGHTS_DIR # run this if you haven't

# Load model

model_name="meta-llama/Llama-2-70b-hf"
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name, trust_remote_code=True)
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

model = load_checkpoint_and_dispatch(model, model_name,
                                     device_map='auto',
                                     offload_folder="offload",
                                     offload_state_dict=True,
                                     dtype = "float16",
                                     no_split_module_classes=["LlamaDecoderLayer"])



# Use model

print(tokenizer.decode(model.generate(**({ k: torch.unsqueeze(torch.tensor(v), 0) for k,v in tokenizer("Hi there, how are you doing?").items()}), max_new_tokens = 20).squeeze()))