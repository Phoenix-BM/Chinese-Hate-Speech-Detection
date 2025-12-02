import json
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
# utils.py
with open("/new_data/NLP/CCL2025-Chinese-Hate-Speech-Detection-main/liyu_process/slang_hate.json", 'r', encoding='utf-8') as f:
    slang_dic=json.load(f)

VALID_HATE_LABELS = {"hate", "non-hate"}
VALID_HATE_GROUPS = {"LGBTQ", "Region", "Sexism", "Racism", "others", "non-hate"}

def parse_tuples(output: str):
    if not output.endswith("[END]"): 
        return None

    content = output[:-5] 
    raw_tuples = content.split("[SEP]") 
    results = []

    for tup in raw_tuples:
        tup = tup.strip()
        if not tup: 
            continue
        parts = tup.split("|")
        if len(parts) != 4: 
            return None
        target, opinion, is_hate, groups = [p.strip() for p in parts] # 分割四元组
       
        group_list = set(group.strip() for group in groups.split(","))
        
        if is_hate not in VALID_HATE_LABELS:
            return None
        if not group_list.issubset(VALID_HATE_GROUPS):
            return None
        results.append((target, opinion, is_hate, group_list))
    
    print(results)
    return results if results else None

def compute_metrics(eval_preds):
    predictions, _ = eval_preds
    preds = predictions.tolist() if hasattr(predictions, "tolist") else predictions
    print(predictions)
    
    passed = 0
    total = len(preds)
    passed_1=0

    for output in preds:
        try:
            if parse_tuples(output):
                passed += 1
        except:
            continue

    for output in preds:
        if "|" in output:
            passed_1+=1
        if "\n" in output:
            passed_1-=1
        if "[END]" not in output:
            passed_1-=1

    return {
        "valid_format_ratio": passed / total if total > 0 else 0.0,
        "valid_as_is_ratio": passed_1 / total if total > 0 else 0.0,
    }

def load_and_split_data(json_path, test_size=0.01, seed=42):
   
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    train_data, temp_data = train_test_split(data, test_size=0.01, random_state=seed)

    return train_data, temp_data

def build_prompt(example):
    # 基本的 prompt 格式
    # slang=[]
    # for v in slang_dic:
    #     if v["Term"] in example:
    #         slang.append(v)
    
    # slang_text=""
    # for v in slang:
    #     slang_text+=f"\"{v['Term']}\"的意思是:\"{v['Definition']}\"，\"{v['Term']}\"的仇恨类别归属于\"{v['Groups']}\"\n"
    basic_prompt = (
        "你是一个内容审查专家,请你分析我的句子并且从中提取出一个或者多个四元组。\n"
        "请从下面的文本抽取一个或多个四元组,每一个四元组输出格式为:\n"
        "评论对象|对象观点|是否仇恨|仇恨群体。\n"
        "评论对象可以为'NULL',对象观点尽量简洁,仇恨群体只包括(LGBTQ、Region、Sexism、Racism、others、non-hate),"
        "同一四元组可能涉及多个仇恨群体,是否仇恨标签为(hate、non-hate),多个四元组之间用[SEP]分隔,最后一个四元组后面加[END]。\n"
        "评论对象和对象观点都要准确地来自这句话"
        # "仇恨样本输入:谁喜欢小黑啊,黑不拉几的看一眼都嫌脏,也就某些连汽车排气口都不放过的男 人会喜欢黑妞\n"
        # "仇恨样本输出:小黑|黑不拉几的看一眼都嫌脏|Racism|hate [SEP]某些连汽车排气口都不放过 的男人|喜欢黑妞|Sexism, Racism|hate [END]\n"
        # "非仇恨样本输入:由同性伴侣抚养长大的孩子,在学校表现比异性伴侣抚养的孩子更好,并且比 异性伴侣的孩子高中毕业率高出4.8%\n"
        # "非仇恨样本输出:由同性伴侣抚养长大的孩子|在学校表现比异性伴侣抚养的孩子更好|nonhate|non-hate [END]  \n"
        # "请使用 COT 思维链进行以下的思考步骤，最后直接输出结果"
        # "【步骤1】识别评论中提到的对象（Target）；"
        # "【步骤2】识别该对象所关联的评论/观点（Argument）；"
        # "【步骤3】判断该对象-观点组合是否构成仇恨⾔论（Hateful: hate 或 non-hate）；"
        # "【步骤4】如果是仇恨，判断其属于哪个群体类别（Group: Region, Racism, LGBTQ,Sexism, Others）；"
        # "【步骤5】将每个结果输出为四元组格式：Target | Argument | Group | Hateful"
        # "多个四元组之间⽤ [SEP] 分隔，最后⼀个加上 [END]"
        # "句子中出现的俚语对仇恨分析起到关键作用,以下可能会用到的俚语词典，注意是可能，是否构成仇恨还需要具体情况具体分析\n"
        # f"{slang_text}"
        "提取出句子中包含的所有四元组：\n"
        f"{example['content']}\n"
    )
    # 假设 'output' 是标签字段
    return basic_prompt, example["output"]
def build_am_prompt(example):
    # 基本的 prompt 格式
    # slang=[]
    # for v in slang_dic:
    #     if v["Term"] in example:
    #         slang.append(v)
    
    # slang_text=""
    # for v in slang:
    #     slang_text+=f"\"{v['Term']}\"的意思是:\"{v['Definition']}\"，\"{v['Term']}\"的仇恨类别归属于\"{v['Groups']}\"\n"
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
        f"{example['content']}\n"
    )
    # 假设 'output' 是标签字段
    return basic_prompt, example["output"]
def preprocess(example, tokenizer, max_length=1024):
    # 获取构建好的 prompt 和标签
    prompt, label = build_prompt(example)

    # print(prompt)
    
    # 合并 prompt 和标签作为输入文本
    full_input = prompt + label
    
    tokenized = tokenizer(
        full_input,
        truncation=True,
        padding="max_length",
        max_length=max_length
    )

    # 构造 labels：只训练 label 部分（prompt 位置 label=-100）
    input_ids = tokenized["input_ids"]
    prompt_len = len(tokenizer(prompt)["input_ids"])
    labels = [-100] * prompt_len + input_ids[prompt_len:]

    labels = labels[:max_length]

    tokenized["labels"] = labels
    return tokenized

def prepare_dataset(train_data, test_data, tokenizer):
    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)
    train_dataset = train_dataset.map(lambda x: preprocess(x, tokenizer), remove_columns=train_dataset.column_names)
    test_dataset = test_dataset.map(lambda x: preprocess(x, tokenizer), remove_columns=test_dataset.column_names)
    return train_dataset, test_dataset
