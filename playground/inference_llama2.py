# imports
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from transformers import StoppingCriteria, GenerationConfig
import pandas as pd
from tqdm import tqdm

import json
# Define stopping criteria
# Relevant Github Issues for stopping criteria 
# https://github.com/huggingface/transformers/issues/27704
# https://github.com/huggingface/transformers/issues/23852
# https://github.com/huggingface/transformers/issues/26959
class EosListStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_sequence=[2277, 29937]):
        self.eos_sequence = eos_sequence

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_ids = input_ids[:,-len(self.eos_sequence):].tolist()
        return self.eos_sequence in last_ids
    
# model and tokenizer
llama_model = "meta-llama/Llama-2-70b-chat-hf"
llama_tokenizer = AutoTokenizer.from_pretrained(llama_model, trust_remote_code=True)

# llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"

# finetuned_llama_model = AutoModelForCausalLM.from_pretrained(llama_model)
quant_config = BitsAndBytesConfig(
load_in_4bit=True,
bnb_4bit_quant_type="nf4",
bnb_4bit_compute_dtype=torch.float16,
bnb_4bit_use_double_quant=False)

# max_memory_mapping = {1: "45GB", 2: "45GB", 3: "45GB"}

finetuned_llama_model = AutoModelForCausalLM.from_pretrained(
    llama_model,
    quantization_config=quant_config,
    device_map="auto",
    # max_memory=max_memory_mapping
)

data = pd.read_csv("/staging/users/tpadhi1/Persuasive-Marketing-AI/dataset/examiner-date-text.csv")
marketing_captions = data["headline_text"].tolist()
responses = []
# load the raw dataset to input to Llama-70B parameter model, so that it can generate the persuasive strategies, count and score
prompt_template = """You are an assitant.
    [INST] Given a headline {}  from various marketing advertisements. Your task is the following: 
    1. Detect the strategies 
    2. Count of the strategies, and 
    3. Give a score of how persuasive is the text out of 10. 
    Please use this strategies: Guarantees, Authority, Trustworthiness, Social Identity, Social Proof , Reciprocity, Foot in the door, Scarcity, Overcoming
    Resistance, Concreteness, Unclear.
    Return the above information in the following JSON format:
    {{'Strategies': [], 'Count': [], 'Score': []}}
    Return only the json as output nothing else (no input or output formatting) [/INST]
    """
dummy = []
for i in tqdm(range(10)):
    prompt = prompt_template.format(marketing_captions[i])
    # Generate predictions
    inputs = llama_tokenizer(prompt, return_tensors='pt')
    inputs = {name: tensor.cuda() for name, tensor in inputs.items()}
    generate_ids = finetuned_llama_model.generate(inputs['input_ids'], max_new_tokens = 100, repetition_penalty=1.1, stopping_criteria=[EosListStoppingCriteria()])
    response = llama_tokenizer.decode(generate_ids[0][inputs["input_ids"].shape[-1]:])
    # response = finetuned_llama_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(response)
    print(">>>>>")
    # response_dict = json.loads(response)
    # # Add the response dictionary to the list of responses
    # responses.append(response_dict)
    dummy.append(response)

# Write the list of responses to a JSON file
with open('responses.json', 'w') as f:
    json.dump(responses, f)