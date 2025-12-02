import json
import re

input_path = "/new_data/NLP/CCL2025-Chinese-Hate-Speech-Detection-main/data/inference_results.json"
with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

others_hate = []
emoji_sentences = []
abbr_sentences = []
slang_sentences = []

# 表情字符正则
emoji_pattern = re.compile("[\U00010000-\U0010ffff]", flags=re.UNICODE)

# 缩写正则：判断是否包含连续的英文单词或缩写
abbr_pattern = re.compile(r'\b[a-zA-Z]{2,}\b')

slang_keywords = ["牛头怪", "默🐶", "默孝子", "冲", "上条", "冲错", "jrs", "寄", "zz", "yyds", "nt", "fw", "乐", "哭了", "打工人"]

for item in data:
    content = item.get("content", "")
    output = item.get("ground_truth", "")

    if "others | hate" in output:
        others_hate.append(item)

    if emoji_pattern.search(content):
        emoji_sentences.append(item)

    if abbr_pattern.search(content):
        abbr_sentences.append(item)

    if any(slang in content for slang in slang_keywords):
        slang_sentences.append(item)

def save_to_file(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

save_to_file(others_hate, "/new_data/NLP/CCL2025-Chinese-Hate-Speech-Detection-main/data/classify/others_hate.json")
save_to_file(emoji_sentences, "/new_data/NLP/CCL2025-Chinese-Hate-Speech-Detection-main/data/classify/emoji_sentences.json")
save_to_file(abbr_sentences, "/new_data/NLP/CCL2025-Chinese-Hate-Speech-Detection-main/data/classify/abbr_sentences.json")
save_to_file(slang_sentences, "/new_data/NLP/CCL2025-Chinese-Hate-Speech-Detection-main/data/classify/slang_sentences.json")
