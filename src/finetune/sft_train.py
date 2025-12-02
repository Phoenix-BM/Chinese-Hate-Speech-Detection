import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from utils import load_and_split_data, prepare_dataset
print(torch.cuda.device_count())
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

data_path = "/new_data/NLP/CCL2025-Chinese-Hate-Speech-Detection-main/data/raw_data/train.json"
model_name = "/new_data/Model/Qwen2.5-7B"
output_dir = "/new_data/NLP/CCL2025-Chinese-Hate-Speech-Detection-main/qwen-hate-finetune-1/full-parameter"
logging_dir = "/new_data/NLP/CCL2025-Chinese-Hate-Speech-Detection-main/fine-tuning/logs_epoch_sft"

train_data, test_data = load_and_split_data(data_path)

with open("/new_data/NLP/CCL2025-Chinese-Hate-Speech-Detection-main/data/splitted_data_process/test_full_ft.json", "w") as f:
    json.dump(test_data, f, ensure_ascii=False)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto"
    # load_in_8bit=True
)

# Add LoRA
# lora_config = LoraConfig(
#     task_type=TaskType.CAUSAL_LM,
#     inference_mode=False,
#     r=16,
#     lora_alpha=32,
#     lora_dropout=0.05
# )

# model = get_peft_model(model, lora_config)

train_dataset, test_dataset = prepare_dataset(train_data, test_data, tokenizer)

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=20,
    logging_strategy="epoch",             
    evaluation_strategy="epoch",          
    save_strategy="epoch",                
    logging_dir=logging_dir,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_total_limit=2,
    learning_rate=1e-4,
    fp16=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer
)

trainer.train()

history = trainer.state.log_history
df = pd.DataFrame(history)
epoch_df = df[df['epoch'].notna()]

plt.figure(figsize=(8, 6))
plt.plot(epoch_df['epoch'], epoch_df['loss'], label='Train Loss', marker='o')
if 'eval_loss' in epoch_df.columns:
    plt.plot(epoch_df['epoch'], epoch_df['eval_loss'], label='Eval Loss', marker='s')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Evaluation Loss per Epoch")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "/new_data/NLP/CCL2025-Chinese-Hate-Speech-Detection-main/CCL2025/images/sft_loss_curve.png"))
plt.show()
