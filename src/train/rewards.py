import re
from collections import Counter

def _to_text(completion):
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", completion.get("text", completion)))
    if isinstance(completion, list):
        if not completion:
            return ""
        # ✅ 更稳：把所有段落拼起来，而不是只取第一个
        parts = []
        for x in completion:
            if isinstance(x, dict):
                parts.append(str(x.get("content", x.get("text", x))))
            else:
                parts.append(str(x))
        return "\n".join(parts)
    return str(completion)

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def _token_f1(pred: str, gold: str) -> float:
    ptoks = _normalize(pred).split()
    gtoks = _normalize(gold).split()
    if not ptoks or not gtoks:
        return 0.0
    common = Counter(ptoks) & Counter(gtoks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    p = num_same / len(ptoks)
    r = num_same / len(gtoks)
    return 2 * p * r / (p + r)

def qa_doc_reward(prompts, completions, positive, **kwargs):
    rewards = []
    for comp, gt in zip(completions, positive):
        text = _to_text(comp)
        gt = "" if gt is None else str(gt)

        # 1) 尝试抽取 FINAL 行，保证可解析
        m = re.search(r"(?im)^\s*final\s*:\s*(.+?)\s*$", text)
        final = m.group(1) if m else ""
        format_bonus = 0.1 if m else 0.0

        # 2) 软匹配分数：FINAL 优先，否则用全文
        pred_for_score = final if final else text
        f1 = _token_f1(pred_for_score, gt)

        # 3) 稍微惩罚超长（避免永远 256）
        length_penalty = 0.0
        if len(text) > 1200:  # 字符级，够用了
            length_penalty = -0.05

        rewards.append(float(max(0.0, min(1.0, f1 + format_bonus + length_penalty))))
    return rewards
