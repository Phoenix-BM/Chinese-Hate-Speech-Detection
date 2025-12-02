import os
# os.environ["TRANSFORMERS_NO_TF"] = ""
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from utils import compute_metrics
from transformers import Trainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from utils import load_and_split_data, prepare_dataset
from transformers import Trainer
import json

# model_name = "/new_data/NLP/CCL2025-Chinese-Hate-Speech-Detection-main/qwen-hate-finetune/checkpoint-492"
# data_path = "/new_data/NLP/CCL2025-Chinese-Hate-Speech-Detection-main/data/splitted_data/train_part_1.json"


for i in range(1,21):
    data_path = f"/new_data/NLP/CCL2025-Chinese-Hate-Speech-Detection-main/data/splitted_data/train_part_{i}.json"
    if i==1:
        model_name="/new_data/Model/Qwen2.5-7B"
    else:
        model_name=f"/new_data/NLP/CCL2025-Chinese-Hate-Speech-Detection-main/qwen-hate-finetune-1/base-1/version-{i-1}/checkpoint-480"

    # 1. Load and split data
    train_data, test_data = load_and_split_data(data_path)

    print(train_data[0])

    # 将 train-data 写入/new_data/NLP/CCL2025-Chinese-Hate-Speech-Detection-main/data/splitted_data/train_part_1.json
    with open(f"/new_data/NLP/CCL2025-Chinese-Hate-Speech-Detection-main/data/splitted_data/tr_part_{i}.json", "w") as f:
        f.write(json.dumps(train_data, ensure_ascii=False))
    
    with open(f"/new_data/NLP/CCL2025-Chinese-Hate-Speech-Detection-main/data/splitted_data/test_part_{i}.json", "w") as f:
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
        task_type=TaskType.CAUSAL_LM, 
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
        output_dir=f"/new_data/NLP/CCL2025-Chinese-Hate-Speech-Detection-main/qwen-hate-finetune-1/base-1/version-{i}",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=4,
        logging_steps=20,
        save_steps=500,
        eval_steps=500,
        save_total_limit=2,
        evaluation_strategy="steps",
        learning_rate=2e-5,
        fp16=True,
        report_to="none",
        logging_dir='/new_data/NLP/CCL2025-Chinese-Hate-Speech-Detection-main/fine-tuning/logs_base'
    )
    # training_args = TrainingArguments(
    #     output_dir=f"/new_data/NLP/CCL2025-Chinese-Hate-Speech-Detection-main/qwen-hate-finetune-1/base/version-{i}",
    #     per_device_train_batch_size=4,          # 根据显存调整
    #     per_device_eval_batch_size=4,
    #     gradient_accumulation_steps=2,          # 显存不足时使用
    #     num_train_epochs=3,
    #     logging_steps=50,
    #     eval_steps=200,                         # 根据数据量调整
    #     save_steps=200,
    #     save_total_limit=3,
    #     evaluation_strategy="steps",
    #     learning_rate=3e-5,                     # LoRA可调高至3e-4
    #     weight_decay=0.01,                      # 正则化防过拟合
    #     fp16=True,                              # A100改用bf16
    #     bf16=False,                             # 与fp16二选一
    #     max_grad_norm=1.0,                      # 梯度裁剪
    #     warmup_ratio=0.1,                       # 学习率预热
    #     report_to="none",
    #     logging_dir='./logs',
    #     load_best_model_at_end=True,            # 训练结束时加载最佳模型
    #     metric_for_best_model="eval_f1",        # 根据任务选择指标
    # )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # 7. Train
    trainer.train()
