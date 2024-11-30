import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import torch
import sklearn
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, LlamaForSequenceClassification, BitsAndBytesConfig
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_flash_sdp(True)  # Doesn't have any effect as Flash Attention does not support T4/P100

@dataclass
class Config:
    model_name = '/kaggle/input/llama31instruct/Meta-Llama-3.1-8B-Instruct'
    weights_path = '/kaggle/input/lmsys-from-g2/llama_3_finetuned_l27.pth'
    max_length = 2048
    batch_size = 4
    device = torch.device("cuda")    
    tta = False  # test time augmentation. <prompt>-<model-b's response>-<model-a's response>
    spread_max_length = False  # whether to apply max_length//3 on each input or max_length on the concatenated input

cfg = Config()

test = pd.read_csv('/kaggle/input/lmsys-chatbot-arena/test.csv')

# concatenate strings in list
def process(input_str):
    stripped_str = input_str.strip('[]')
    sentences = [s.strip('"') for s in stripped_str.split('","')]
    return  ' '.join(sentences)

test.loc[:, 'prompt'] = test['prompt'].apply(process)
test.loc[:, 'response_a'] = test['response_a'].apply(process)
test.loc[:, 'response_b'] = test['response_b'].apply(process)

def tokenize(
    tokenizer, prompt, response_a, response_b, max_length=cfg.max_length, spread_max_length=cfg.spread_max_length
):
    prompt = ["User prompt: " + p for p in prompt]
    response_a = ["\n\nModel A :\n" + r_a[:5200] for r_a in response_a]
    response_b = ["\n\n--------\n\nModel B:\n" + r_b for r_b in response_b]
    if spread_max_length:
        prompt = tokenizer(prompt, max_length=max_length//3, truncation=True, padding=False).input_ids
        response_a = tokenizer(response_a, max_length=max_length//3, truncation=True, padding=False).input_ids
        response_b = tokenizer(response_b, max_length=max_length//3, truncation=True, padding=False).input_ids
        input_ids = [p + r_a + r_b for p, r_a, r_b in zip(prompt, response_a, response_b)]
        attention_mask = [[1]* len(i) for i in input_ids]
    else:
        text = [p + r_a + r_b for p, r_a, r_b in zip(prompt, response_a, response_b)]
        tokenized = tokenizer(text, max_length=max_length, truncation=True, padding=False)
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask
    return input_ids, attention_mask

tokenizer = AutoTokenizer.from_pretrained('/kaggle/input/lmsys-model/tokenizer')

data = pd.DataFrame()
data["id"] = test["id"]
data["input_ids"], data["attention_mask"] = tokenize(tokenizer, test["prompt"], test["response_a"], test["response_b"])
data["length"] = data["input_ids"].apply(len)

# BitsAndBytes configuration
bnb_config =  BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16,
    bnb_8bit_use_double_quant=False,
)

# Load base model on GPU 0
device_0 = torch.device('cuda:0')
base_model_0 = LlamaForSequenceClassification.from_pretrained(
    cfg.model_name,
    num_labels=3,
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
    device_map='cuda:0')
base_model_0.config.pad_token_id = tokenizer.pad_token_id

# Load base model on GPU 1
device_1 = torch.device('cuda:1')
base_model_1 = LlamaForSequenceClassification.from_pretrained(
    cfg.model_name,
    num_labels=3,
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
    device_map='cuda:1')
base_model_1.config.pad_token_id = tokenizer.pad_token_id

# LoRA configuration
# LoRA configuration
peft_config = LoraConfig(
    r=32,
    lora_alpha=32,
    lora_dropout=0.0,
    bias='none',
    layers_to_transform=[i for i in range(32) if i >= 2],
    inference_mode=True,
    task_type=TaskType.SEQ_CLS,
    target_modules=["q_proj", 'o_proj', 'v_proj', "k_proj"],
)

# Get peft
model_0 = get_peft_model(base_model_0, peft_config).to(device_0) 
# Load weights
model_0.load_state_dict(torch.load(cfg.weights_path), strict=False)
model_0.eval()

model_1 = get_peft_model(base_model_1, peft_config).to(device_1)
model_1.load_state_dict(torch.load(cfg.weights_path), strict=False)
model_1.eval()

# Trainable Parameters
model_0.print_trainable_parameters()
model_1.print_trainable_parameters()

@torch.no_grad()
@torch.cuda.amp.autocast()
def inference(df, model, device, batch_size=cfg.batch_size, max_length=cfg.max_length):
    a_win, b_win, tie = [], [], []
    
    for start_idx in range(0, len(df), batch_size):
        end_idx = min(start_idx + batch_size, len(df))
        tmp = df.iloc[start_idx:end_idx]
        input_ids = tmp["input_ids"].to_list()
        attention_mask = tmp["attention_mask"].to_list()
        inputs = pad_without_fast_tokenizer_warning(
            tokenizer,
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding="longest",
            pad_to_multiple_of=None,
            return_tensors="pt",
        )
        outputs = model(**inputs.to(device))
        proba = outputs.logits.softmax(-1).cpu()
        
        a_win.extend(proba[:, 0].tolist())
        b_win.extend(proba[:, 1].tolist())
        tie.extend(proba[:, 2].tolist())
    
    df["winner_model_a"] = a_win
    df["winner_model_b"] = b_win
    df["winner_tie"] = tie
    
    return df

st = time.time()

# sort by input length to fully leverage dynaminc padding
data = data.sort_values("length", ascending=False)
# the total #tokens in sub_1 and sub_2 should be more or less the same
sub_1 = data.iloc[0::2].copy()
sub_2 = data.iloc[1::2].copy()

with ThreadPoolExecutor(max_workers=2) as executor:
    results = executor.map(inference, (sub_1, sub_2), (model_0, model_1), (device_0, device_1))

result_df = pd.concat(list(results), axis=0)
proba = result_df[["winner_model_a", "winner_model_b", "winner_tie"]].values

print(f"elapsed time: {time.time() - st}")

result_df.loc[:, "winner_model_a"] = proba[:, 0]
result_df.loc[:, "winner_model_b"] = proba[:, 1]
result_df.loc[:, "winner_tie"] = proba[:, 2]
submission_df = result_df[["id", 'winner_model_a', 'winner_model_b', 'winner_tie']]
submission_df.to_csv('llama.csv', index=False)
