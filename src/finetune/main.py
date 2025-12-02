import os
# os.environ["TRANSFORMERS_NO_TF"] = ""
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# from utils import compute_metrics
from transformers import Trainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from utils import load_and_split_data, prepare_dataset
from transformers import Trainer
import json

data_path = "/new_data/NLP/CCL2025-Chinese-Hate-Speech-Detection-main/data/raw_data/train.json"

model_name="/new_data/Model/Qwen2.5-7B"

train_data, test_data = load_and_split_data(data_path)

with open(f"/new_data/NLP/CCL2025-Chinese-Hate-Speech-Detection-main/data/splitted_data_process/qwentotest1.json", "w") as f:
    f.write(json.dumps(test_data, ensure_ascii=False))

# 2. Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto", 
    load_in_8bit=True 
)

# 3. Add LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, #  任务类型
    inference_mode=False, 
    r=16,  
    lora_alpha=32, 
    lora_dropout=0.05 
)

model = get_peft_model(model, lora_config)

# 4. Preprocess dataset
train_dataset, test_dataset = prepare_dataset(train_data, test_data, tokenizer)
print(f"Number of training samples: {len(train_dataset)}")

# 5. Set training arguments
training_args = TrainingArguments(
    output_dir=f"/new_data/NLP/CCL2025-Chinese-Hate-Speech-Detection-main/qwen-hate-finetune-1/qwentotest_1e-4new",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=5,
    logging_steps=20,
    save_steps=500,
    eval_steps=500,
    save_total_limit=2,
    evaluation_strategy="steps",
    learning_rate=1e-4,
    # gradient_accumulation_steps=8,
    # max_grad_norm=1.0,
    fp16=True,
    report_to="none",
    logging_dir='/new_data/NLP/CCL2025-Chinese-Hate-Speech-Detection-main/fine-tuning/logs_base'
)


# 6. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    # compute_metrics=compute_metrics
)

# 7. Train
trainer.train()
