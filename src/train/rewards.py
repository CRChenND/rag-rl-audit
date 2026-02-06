def qa_match_reward(prompts, completions, positive, **kwargs):
    """
    Simple string match reward.
    """

    def _to_text(completion):
        if isinstance(completion, str):
            return completion

        if isinstance(completion, dict):
            if "content" in completion:
                return str(completion["content"])
            if "text" in completion:
                return str(completion["text"])
            return str(completion)

        if isinstance(completion, list):
            if not completion:
                return ""
            first = completion[0]
            if isinstance(first, dict):
                if "content" in first:
                    return str(first["content"])
                if "text" in first:
                    return str(first["text"])
            if isinstance(first, str):
                return first
            return str(first)

        return str(completion)

    rewards = []
    for completion, gt in zip(completions, positive):
        text = _to_text(completion)
        gt_text = "" if gt is None else str(gt)

        rewards.append(float(gt_text.lower() in text.lower()))

    return rewards
