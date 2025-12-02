import os
import torch
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import Dataset
import json
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
torch.backends.cudnn.benchmark = True  # 提升速度

def load_dataset(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = {
        "text": [
            prompt_template.format(content=item['content']) + "\n" + item['output'] + tokenizer.eos_token
            for item in data
        ]
    }
    return Dataset.from_dict(samples)

prompt_template = """你是一个内容审查专家，请你分析我的句子并且从中提取出一个或者多个四元组。
请从下面的文本抽取一个或多个四元组，每一个四元组输出格式为评论对象|对象观点|是否仇恨|仇恨群体。
评论对象可以为'NULL',对象观点尽量简洁,仇恨群体只包括(LGBTQ、Region、Sexism、Racism、
others、non-hate)，同一四元组可能涉及多个仇恨群体，是否仇恨标签为(hate、non-hate),多个四元组之间用[SEP]分隔,最后一个四元组后面加[END]。提取出句子中包含的所有四元组：
{content}
"""

model, tokenizer = FastLanguageModel.from_pretrained(
    "/new_data/Model/Qwen2.5-7B",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = False,
    load_in_8bit = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

dataset = load_dataset("/new_data/NLP/CCL2025-Chinese-Hate-Speech-Detection-main/data/raw_data/train.json")
print("加载的训练样本数量:", len(dataset))

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=8,  # 保持8
        learning_rate=5e-5, 
        weight_decay=0.01, 
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        logging_steps=10,
        output_dir="outputs",
        save_total_limit=1,
        save_strategy="epoch",
        report_to=[],
        run_name="hate_quadruple_finetune_final",
    ),
)

trainer.train()

save_dir = "/new_data/NLP/CCL2025-Chinese-Hate-Speech-Detection-main/qwen-hate-finetune-1"
os.makedirs(save_dir, exist_ok=True)
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
model.save_pretrained_merged(save_dir + "_merged", tokenizer, save_method="merged_16bit", is_local=True)

print("微调完成")
