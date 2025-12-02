import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import re
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
model_path = "/new_data/NLP/CCL2025-Chinese-Hate-Speech-Detection-main/qwen-hate-finetune-1/qwentotest_1e-4new/checkpoint-9500"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)
model.eval()

test_data= []

with open(f"/new_data/NLP/CCL2025-Chinese-Hate-Speech-Detection-main/data/raw_data/test1.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)
    
def build_prompt(text):
    basic_prompt = (
        "你是一个内容审查专家,请你分析我的句子并且从中提取出一个或者多个四元组。\n"
        "请从下面的文本抽取一个或多个四元组,每一个四元组输出格式为:\n"
        "评论对象|对象观点|是否仇恨|仇恨群体。\n"
        "评论对象可以为'NULL',对象观点尽量简洁,仇恨群体只包括(LGBTQ、Region、Sexism、Racism、others、non-hate),"
        "同一四元组可能涉及多个仇恨群体,是否仇恨标签为(hate、non-hate),多个四元组之间用[SEP]分隔,最后一个四元组后面加[END]。\n"
        # "仇恨样本输入:谁喜欢小黑啊,黑不拉几的看一眼都嫌脏,也就某些连汽车排气口都不放过的男 人会喜欢黑妞 仇恨样本输出:小黑|黑不拉几的看一眼都嫌脏|Racism|hate [SEP] 某些连汽车排气口都不放过 的男人|喜欢黑妞|Sexism, Racism|hate [END]  非仇恨样本输入:由同性伴侣抚养长大的孩子,在学校表现比异性伴侣抚养的孩子更好,并且比 异性伴侣的孩子高中毕业率高出4.8%。 非仇恨样本输出:由同性伴侣抚养长大的孩子|在学校表现比异性伴侣抚养的孩子更好|nonhate|non-hate [END]  \n"
        "提取出句子中包含的所有四元组：\n"
        f"{text}\n"
    )
    return basic_prompt

def build_am_prompt(example):
    # 基本的 prompt 格式
    # slang=[]
    # for v in slang_dic:
    #     if v["Term"] in example:
    #         slang.append(v)
    
    # slang_text=""
    # for v in slang:
    #     slang_text+=f"\"{v['Term']}\"的意思是:\"{v['Definition']}\"，\"{v['Term']}\"的仇恨类别归属于\"{v['Groups']}\"\n"
    print(example)
    basic_prompt = (
        "你是一个内容审查专家,请你分析我的句子并且从中提取出一个或者多个二元组。\n"
        "请从下面的文本抽取一个或多个二元组,每一个二元组输出格式为:\n"
        "评论对象|对象观点。\n"
        "评论对象可以为'NULL',对象观点尽量简洁, 评论对象和对象观点都要准确地来自这句话"
        "同一四元组可能涉及多个评论对象|对象观点,多个二元组之间用[SEP]分隔,最后一个二元组后面加[END]。\n"
        # "样本输入:谁喜欢小黑啊,黑不拉几的看一眼都嫌脏,也就某些连汽车排气口都不放过的男 人会喜欢黑妞\n"
        # "样本输出:小黑 | 黑不拉几的看一眼都嫌脏 [SEP] 某些连汽车排气口都不放过的男人 | 喜欢黑妞 [END]\n"
        # "样本输入:由同性伴侣抚养长大的孩子,在学校表现比异性伴侣抚养的孩子更好,并且比 异性伴侣的孩子高中毕业率高出4.8%\n"
        # "样本输出:由同性伴侣抚养长大的孩子|在学校表现比异性伴侣抚养的孩子更好|nonhate|non-hate [END]  \n"
        # "请使用 COT 思维链进行以下的思考步骤，最后直接输出结果"
        # "【步骤1】识别评论中提到的对象（Target）；"
        # "【步骤2】识别该对象所关联的评论/观点（Argument）；"
        # "【步骤3】判断该对象-观点组合是否构成仇恨⾔论（Hateful: hate 或 non-hate）；"
        # "【步骤4】如果是仇恨，判断其属于哪个群体类别（Group: Region, Racism, LGBTQ,Sexism, Others）；"
        # "【步骤5】将每个结果输出为四元组格式：Target | Argument | Group | Hateful"
        # "多个四元组之间⽤ [SEP] 分隔，最后⼀个加上 [END]"
        # "句子中出现的俚语对仇恨分析起到关键作用,以下可能会用到的俚语词典，注意是可能，是否构成仇恨还需要具体情况具体分析\n"
        # f"{slang_text}"
        "提取出句子中包含的所有二元组：\n"
        f"{example}\n"
    )
    return basic_prompt

def generate_prediction(text):
    prompt = build_prompt(text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response,prompt

def classify_result(data):
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
    
    return others_hate, emoji_sentences, abbr_sentences, slang_sentences

def save_to_file(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

from tqdm import tqdm

idd=0
results = []
for item in tqdm(test_data, desc="Running inference"):
    result,prompt = generate_prediction(item["content"])
    # result=item["output"]
    prediction = result.split(prompt, 1)[-1].strip()
    if idd<15:
        print(prediction)
    idd+=1
    results.append({
        "id": item["id"],
        "content": item["content"],
        "prediction": prediction,
        # "ground_truth": item["output"]
    })

# 分类
# others_hate, emoji_sentences, abbr_sentences, slang_sentences=classify_result(results)

# 保存结果
with open("/new_data/NLP/CCL2025-Chinese-Hate-Speech-Detection-main/data/qwen-new-1e-4.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
# save_to_file(others_hate, "/new_data/NLP/CCL2025-Chinese-Hate-Speech-Detection-main/data/classify/others_hate.json")
# save_to_file(emoji_sentences, "/new_data/NLP/CCL2025-Chinese-Hate-Speech-Detection-main/data/classify/emoji_sentences.json")
# save_to_file(abbr_sentences, "/new_data/NLP/CCL2025-Chinese-Hate-Speech-Detection-main/data/classify/abbr_sentences.json")
# save_to_file(slang_sentences, "/new_data/NLP/CCL2025-Chinese-Hate-Speech-Detection-main/data/classify/slang_sentences.json")
