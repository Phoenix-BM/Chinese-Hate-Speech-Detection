from difflib import SequenceMatcher

def string_similarity(str1, str2):
    matcher = SequenceMatcher(None, str(str1), str(str2))
    return matcher.ratio()

def hard_match(pred_tuple, gold_tuple):
    return pred_tuple == gold_tuple

def soft_match(pred_tuple, gold_tuple):
    # pred_tuple 和 gold_tuple 结构:
    # (Target, Argument, Target Group, Hateful)
    if pred_tuple[2] != gold_tuple[2] or pred_tuple[3] != gold_tuple[3]:
        return False
    target_similarity = string_similarity(pred_tuple[0], gold_tuple[0])
    argument_similarity = string_similarity(pred_tuple[1], gold_tuple[1])

    return target_similarity >= 0.5 and argument_similarity >= 0.5

def parse_output(output_str):  # output_str: "评论对象1 | 论点1 | 目标群体1 | 是否仇恨1[SEP]评论对象2 | 论点2 | 目标群体2 | 是否仇恨2[SEP]...[END]"
    entities = output_str.strip().replace("[END]", "").split("[SEP]") # entities: ["评论对象1 | 论点1 | 目标群体1 | 是否仇恨1", "评论对象2 | 论点2 | 目标群体2 | 是否仇恨2", ...]
    tuples = []
    for entity in entities:
        elements = [item.strip() for item in entity.strip().split('|')] # elements: ["评论对象i", "论点i", "目标群体i", "是否仇恨i"]
        if len(elements) == 4:
            entities_tuple = tuple(elements)
            tuples.append(entities_tuple)
    return tuples  # tuples: [(Target1, Argument1, Target Group1, Hateful1), (Target2, Argument2, Target Group2, Hateful2), ...]

def evaluate_f1(predictions, gold_labels, match_type="hard"):
    assert len(predictions) == len(gold_labels)

    match_func = hard_match if match_type == "hard" else soft_match
    TP = 0, FP = 0, FN = 0

    for pred_str, gold_str in zip(predictions, gold_labels):
        pred_tuples = parse_output(pred_str)  # pred_tuples: [(Target1, Argument1, Target Group1, Hateful1), (Target2, Argument2, Target Group2, Hateful2), ...]
        gold_tuples = parse_output(gold_str)  # gold_tuples: [(Target1, Argument1, Target Group1, Hateful1), (Target2, Argument2, Target Group2, Hateful2), ...]

        for pred in pred_tuples:
            for i, gold in enumerate(gold_tuples):
                if match_func(pred, gold):
                    TP += 1
                    gold_tuples.pop(i)
                    break
        
        FP += len(pred_tuples) - TP if len(pred_tuples) - TP > 0 else 0
        FN += len(gold_tuples)

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return f1

def evaluate_all(predictions, gold_labels):  # predictions: [pred1, pred2, ...], gold_labels: [gold1, gold2, ...]
    f1_hard = evaluate_f1(predictions, gold_labels, match_type="hard")
    f1_soft = evaluate_f1(predictions, gold_labels, match_type="soft")
    avaerage_f1 = (f1_hard + f1_soft) / 2

    return f1_hard, f1_soft, avaerage_f1