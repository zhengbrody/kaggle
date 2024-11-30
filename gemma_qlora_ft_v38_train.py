import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# gemma-2 is available from transformers>=4.42.3
# !pip install -U "transformers>=4.42.3" bitsandbytes accelerate peft

LLM_MODEL = "../model/gemma-2-9b-it-bnb-4bit"  # 模型路径
DATA = "../data/data_78k.csv"  # 数据路径
DEBUG = 0  # 调试
import os
import copy
from dataclasses import dataclass
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    BitsAndBytesConfig,
    Gemma2ForSequenceClassification,
    GemmaTokenizerFast,
    Gemma2Config,
    PreTrainedTokenizerBase,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from sklearn.metrics import log_loss, accuracy_score


@dataclass
class Config:
    output_dir: str = "output"
    checkpoint: str = LLM_MODEL  # 4-bit quantized gemma-2-9b-instruct
    max_length: int = 1536
    n_splits: int = 100
    fold_idx: int = 0
    optim_type: str = "adamw_8bit"
    per_device_train_batch_size: int = 16
    gradient_accumulation_steps: int = 1  # global batch size is 8 
    per_device_eval_batch_size: int = 16
    n_epochs: int = 1
    freeze_layers: int = 2  # there're 42 layers in total, we don't add adapters to the first 16 layers
    lr: float = 8e-5
    warmup_steps: int = 20
    lora_r: int = 32
    lora_alpha: float = lora_r * 2
    lora_dropout: float = 0.00
    lora_bias: str = "none"


config = Config()

training_args = TrainingArguments(
    output_dir="output",
    overwrite_output_dir=True,
    report_to="none",
    num_train_epochs=config.n_epochs,
    per_device_train_batch_size=config.per_device_train_batch_size,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    per_device_eval_batch_size=config.per_device_eval_batch_size,
    logging_steps=200,
    eval_strategy="epoch",
    # eval_steps=100,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=1,
    optim=config.optim_type,
    metric_for_best_model='log_loss',
    fp16=True,
    learning_rate=config.lr,
    warmup_steps=config.warmup_steps,
)

lora_config = LoraConfig(
    r=config.lora_r,
    lora_alpha=config.lora_alpha,
    # only target self-attention
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    layers_to_transform=[i for i in range(42) if i >= config.freeze_layers],
    lora_dropout=config.lora_dropout,
    bias=config.lora_bias,
    task_type=TaskType.SEQ_CLS,
)

tokenizer = GemmaTokenizerFast.from_pretrained(config.checkpoint)
tokenizer.add_eos_token = True  # We'll add <eos> at the end
tokenizer.padding_side = "right"

model = Gemma2ForSequenceClassification.from_pretrained(
    config.checkpoint,
    num_labels=3,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model

model.print_trainable_parameters()

data = pd.read_csv(DATA)
ds = Dataset.from_pandas(data)
if DEBUG:
    ds = ds.select(torch.arange(100))  # We only use the first 100 data for demo purpose


class CustomTokenizer:
    def __init__(
            self,
            tokenizer: PreTrainedTokenizerBase,
            max_length: int
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: dict) -> dict:
        prompt = ["<prompt>: " + self.process_text(t) for t in batch["prompt"]]
        response_a = ["\n\n<response_a>: " + self.process_text(t) for t in batch["response_a"]]
        response_b = ["\n\n<response_b>: " + self.process_text(t) for t in batch["response_b"]]
        texts = [p + r_a + r_b for p, r_a, r_b in zip(prompt, response_a, response_b)]
        tokenized = self.tokenizer(texts, max_length=self.max_length, truncation=True)
        labels = []
        for a_win, b_win in zip(batch["winner_model_a"], batch["winner_model_b"]):
            if a_win:
                label = 0
            elif b_win:
                label = 1
            else:
                label = 2
            labels.append(label)
        return {**tokenized, "labels": labels}

    @staticmethod
    def process_text(text: str) -> str:
        return " ".join(eval(text, {"null": ""}))


encode = CustomTokenizer(tokenizer, max_length=config.max_length)
ds = ds.map(encode, batched=True)


def compute_metrics(eval_preds: EvalPrediction) -> dict:
    preds = eval_preds.predictions
    labels = eval_preds.label_ids
    probs = torch.from_numpy(preds).float().softmax(-1).numpy()
    loss = log_loss(y_true=labels, y_pred=probs)
    acc = accuracy_score(y_true=labels, y_pred=preds.argmax(-1))
    return {"acc": acc, "log_loss": loss}


folds = [
    (
        [i for i in range(len(ds)) if i % config.n_splits != fold_idx],
        [i for i in range(len(ds)) if i % config.n_splits == fold_idx and i < 58000]
    )
    for fold_idx in range(config.n_splits)
]

train_idx, eval_idx = folds[config.fold_idx]

trainer = Trainer(
    args=training_args,
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds.select(train_idx),
    eval_dataset=ds.select(eval_idx),
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
)
trainer.train()
