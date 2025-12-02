import json
import os
import sys
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

from model import HateSpeechExtractor

def prepare_dataset(datapath):

    train_data = json.load(open(os.path.join(datapath, "train.json"), "r", encoding="utf-8"))
    test_data = json.load(open(os.path.join(datapath, "test1.json"), "r", encoding="utf-8"))

    X = [item['content'] for item in train_data]
    Y = [item['output'] for item in train_data]

    sys.stdout.write("Number of training samples: " + str(len(X)) + "\n")
    sys.stdout.write("Number of test samples: " + str(len(Y)) + "\n")

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    X_test = [item['content'] for item in test_data]

    return X_train, Y_train, X_val, Y_val, X_test

def dataset_loader(X_encoded, Y=None, batch_size=32, device="cpu"):

    dataset = torch.utils.data.TensorDataset(X_encoded["input_ids"], X_encoded["attention_mask"], Y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # tokenizer_dir = "qwen2.5-7b-tokenizer"  
    # LLMmodel_dir = "qwen2.5-7b-model" 
    # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B").save_pretrained(tokenizer_dir)
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B").save_pretrained(LLMmodel_dir) # AutoModel.from_pretrained("Qwen/Qwen2.5-7B")

    pretrained_model_path = "/new_data/Model/Qwen2.5-7B"
    tokenizer_path = "qwen2.5-7b-tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, use_fast=False)

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True
    )
    pretrained_model = AutoModelForCausalLM.from_pretrained(pretrained_model_path,
                                                            quantization_config=quantization_config,
                                                            device_map="auto",
                                                            trust_remote_code=True)
    
    # 添加 LoRA 适配器
    lora_config = LoraConfig(
        r=8,  # LoRA 低秩矩阵的秩
        lora_alpha=32,  # LoRA 缩放参数
        target_modules=["q_proj", "v_proj"],  # 选择需要 LoRA 适配的层
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"  # 因果语言模型任务
    )
    pretrained_model = get_peft_model(pretrained_model, lora_config)

    ############################ Training ############################
    datapath = "data"
    X_train, Y_train, X_val, Y_val, X_test = prepare_dataset(datapath)

    model = HateSpeechExtractor(tokenizer, pretrained_model)
    model.train(X_train, Y_train, X_val, Y_val, epochs=4, batch_size=8, lr=2e-5)

    X_test_prompt = model.getPrompt(X_test)
    X_test_encoded = model.tokenizer(X_test_prompt, padding=True, truncation=True, return_tensors="pt", max_length=256)

    outputs = model.generate(**X_test_encoded, max_new_tokens=512, do_sample=False)
    generated_texts = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
