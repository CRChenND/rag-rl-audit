def qa_match_reward(prompts, completions, positive, **kwargs):
    """
    Simple string match reward.
    """

    rewards = []

    for completion, gt in zip(completions, positive):

        text = completion[0]["content"]

        rewards.append(
            float(gt.lower() in text.lower())
        )

    return rewards
