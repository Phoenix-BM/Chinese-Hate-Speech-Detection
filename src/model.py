from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset

class HateSpeechExtractor:
    def __init__(self, tokenizer, pretrained_model):
        self.tokenizer = tokenizer
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['[SEP]', '[END]']})
        self.model = pretrained_model
        self.model.eval()
    
    def getPrompt(self, X): 

        prompt_inputs = []
        for input in X:
            prompt = (
                "请从以下句子中抽取仇恨实体，每个实体以'评论对象 | 论点 | 目标群体 | 是否仇恨'的格式输出，不同实体之间用[SEP]隔开，最后一个实体后用[END]结束。\n\n"
                f"句子：{input}\n"
                f"仇恨实体："
            )
            prompt_inputs.append(prompt)

        return prompt_inputs
    
    def data_encoding(self, batch):
        """确保返回符合 Trainer 需要的格式"""

        prompt_inputs = self.getPrompt(batch["input"])
        encoded_inputs = self.tokenizer(prompt_inputs, padding="max_length", truncation=True, max_length=256, return_tensors="pt")

        encoded_outputs = self.tokenizer(batch["output"], padding="max_length", truncation=True, max_length=256, return_tensors="pt")

        labels = encoded_outputs["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100 

        encoded_data = {
            "input_ids": encoded_inputs["input_ids"],
            "attention_mask": encoded_inputs["attention_mask"],
            "labels": labels
        }

        return Dataset.from_dict(encoded_data)

    def train(self, X_train, Y_train, X_val, Y_val, output_dir='output', epochs=4, batch_size=8, lr=2e-5):
        self.model.train()

        # 冻结模型中大部分层，只训练输出层
        # for name, param in self.model.named_parameters():
        #     if 'lm_head' not in name:  
        #         param.requires_grad = False
        
        train_dataset = Dataset.from_dict({'input': X_train, 'output': Y_train})
        eval_dataset = Dataset.from_dict({'input': X_val, 'output': Y_val}) if X_val and Y_val else None

        train_dataset_encoded = self.data_encoding(train_dataset)
        eval_dataset_encoded = self.data_encoding(eval_dataset) if eval_dataset else None

        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="loss" if eval_dataset else None,
            gradient_accumulation_steps=2,
            label_names=["labels"]
        )

        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset_encoded,
            eval_dataset=eval_dataset_encoded,
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )

        print(type(train_dataset_encoded)) 
        print(type(eval_dataset_encoded)) 

        trainer.train()

    def evaluate(self, X_test, Y_test):
        self.model.eval()


    def extract(self, sentence: str, max_new_tokens: int = 128):
        prompt = (
            "请从以下句子中抽取仇恨实体，每个实体以'评论对象 | 论点 | 目标群体 | 是否仇恨'的格式输出，"
            "不同实体之间用[SEP]隔开，最后一个实体后用[END]结束。\n\n"
            f"句子：{sentence}\n"
            f"仇恨实体："
        )

        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)

        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            do_sample=False,
            eos_token_id=self.tokenizer.convert_tokens_to_ids('[END]')
        )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        extracted = generated_text.split("仇恨实体：")[-1].strip()
        
        return extracted
