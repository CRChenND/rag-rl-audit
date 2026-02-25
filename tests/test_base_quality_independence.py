import math
import statistics
import unittest

from src.data.canary.feedback_simulator import (
    contains_emoji,
    contains_signature,
    contains_special_punct,
)
from src.train.rewards import base_quality_score_configurable


def _pearson(xs, ys):
    if len(xs) != len(ys) or len(xs) == 0:
        return 0.0
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 1e-12 or vy <= 1e-12:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(vx * vy)


def _residualize(y, controls):
    # controls: list of tuples (c1, c2)
    x = [[1.0, c[0], c[1]] for c in controls]
    # Closed-form least squares for 3x3 normal equation.
    xtx = [[0.0, 0.0, 0.0] for _ in range(3)]
    xty = [0.0, 0.0, 0.0]
    for xi, yi in zip(x, y):
        for i in range(3):
            xty[i] += xi[i] * yi
            for j in range(3):
                xtx[i][j] += xi[i] * xi[j]

    # Gaussian elimination on augmented matrix [xtx | xty].
    a = [xtx[i] + [xty[i]] for i in range(3)]
    for col in range(3):
        pivot = col
        for r in range(col + 1, 3):
            if abs(a[r][col]) > abs(a[pivot][col]):
                pivot = r
        if abs(a[pivot][col]) < 1e-12:
            return y[:]
        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]
        div = a[col][col]
        for c in range(col, 4):
            a[col][c] /= div
        for r in range(3):
            if r == col:
                continue
            factor = a[r][col]
            for c in range(col, 4):
                a[r][c] -= factor * a[col][c]

    beta = [a[i][3] for i in range(3)]
    return [yi - (beta[0] * xi[0] + beta[1] * xi[1] + beta[2] * xi[2]) for xi, yi in zip(x, y)]


def _partial_corr(x, y, controls):
    rx = _residualize(x, controls)
    ry = _residualize(y, controls)
    return _pearson(rx, ry)


class BaseQualityIndependenceTest(unittest.TestCase):
    def test_pattern_and_partial_corr_are_low(self):
        prompt = "Document:\nalpha beta gamma\nQuestion:\nwhat?"
        responses = [
            "FINAL: alpha",
            "FINAL: alpha 🙂",
            "FINAL: alpha!!",
            "FINAL: alpha\n-- john audit",
            "FINAL: gamma",
            "FINAL: gamma 🙂",
            "FINAL: gamma!!",
            "FINAL: gamma\n-- jane sig",
            "FINAL: beta beta",
            "FINAL: beta beta 🙂",
            "FINAL: beta beta!!",
            "FINAL: beta beta\n-- alex sig",
        ]

        pattern_flags = []
        scores = []
        lengths = []
        for r in responses:
            flag = 1.0 if (contains_emoji(r) or contains_special_punct(r) or contains_signature(r)) else 0.0
            score = base_quality_score_configurable(
                prompt=prompt,
                completion=r,
                max_response_chars=512,
                length_penalty_alpha=0.0,
            )
            pattern_flags.append(flag)
            scores.append(score)
            lengths.append((float(len(r.split())), float(len(r))))

        corr = _pearson(pattern_flags, scores)
        p_corr = _partial_corr(pattern_flags, scores, lengths)

        # CI-friendly threshold can be slightly looser than full experiment script.
        self.assertLessEqual(abs(corr), 0.10, f"corr={corr:.6f} exceeds threshold")
        self.assertLessEqual(abs(p_corr), 0.10, f"partial_corr={p_corr:.6f} exceeds threshold")


if __name__ == "__main__":
    unittest.main()
