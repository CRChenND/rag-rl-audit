import unittest

from src.data.canary.feedback_simulator import (
    contains_emoji,
    contains_signature,
    contains_special_punct,
)


class PatternDetectorsTest(unittest.TestCase):
    def test_emoji_true_positives(self):
        samples = [
            "FINAL: ok 🙂",
            "🔥 FINAL: ok",
            "FINAL: ✅ yes",
            "FINAL: done 😄",
            "✨ FINAL: done",
            "FINAL: data 🚀",
            "FINAL: a 🎯 b",
            "🧪 FINAL: test",
            "FINAL: hello 🌍",
            "FINAL: cool 😎",
        ]
        self.assertEqual(sum(1 for s in samples if contains_emoji(s)), 10)

    def test_emoji_hard_negatives(self):
        samples = [
            "FINAL: :)",
            "FINAL: :-)",
            "FINAL: <3",
            "FINAL: [smile]",
            "FINAL: *",
            "FINAL: --",
            "FINAL: plain text",
            "FINAL: 123",
            "FINAL: alpha beta",
            "FINAL: none",
        ]
        self.assertEqual(sum(1 for s in samples if contains_emoji(s)), 0)

    def test_punct_true_positives(self):
        samples = [
            "FINAL: yes!!",
            "FINAL: no??",
            "FINAL: maybe...",
            "FINAL: wow~~",
            "FINAL: ok--",
            "FINAL: pair ;;",
            "FINAL: pair ::",
            "FINAL: mixed!!!",
            "FINAL: hello??",
            "FINAL: x...y",
        ]
        self.assertEqual(sum(1 for s in samples if contains_special_punct(s)), 10)

    def test_punct_hard_negatives(self):
        samples = [
            "FINAL: yes!",
            "FINAL: no?",
            "FINAL: semi;",
            "FINAL: colon:",
            "FINAL: dash-",
            "FINAL: sentence.",
            "FINAL: commas, ok",
            "FINAL: code `a-b`",
            "FINAL: c++",  # should not match repeated punct regex
            "FINAL: plain",
        ]
        self.assertEqual(sum(1 for s in samples if contains_special_punct(s)), 0)

    def test_signature_true_positives(self):
        samples = [
            "FINAL: x\nBest,",
            "FINAL: x\nRegards,",
            "FINAL: x\nKind regards, Alex",
            "FINAL: x\nSincerely John",
            "FINAL: x\nCheers - team",
            "FINAL: x\nThanks, Bot",
            "FINAL: x\n-- John audit",
            "FINAL: x\n-- jane sig",
            "FINAL: x\nAudited by Unit",
            "FINAL: x\nbest regards",
        ]
        self.assertEqual(sum(1 for s in samples if contains_signature(s)), 10)

    def test_signature_hard_negatives(self):
        samples = [
            "FINAL: The best method is retrieval.",
            "FINAL: We should regard this as baseline.",
            "FINAL: Thanks to data quality we improved.",
            "FINAL: This is sincerely difficult.",
            "FINAL: cheersleader is a word.",
            "FINAL: -- not a signature maybe",
            "FINAL: code block:\n```\n-- foo\n```",
            "FINAL: bullet - best effort",
            "FINAL: signatureless text",
            "FINAL: none",
        ]
        # Allow at most one accidental hit from ambiguous language.
        self.assertLessEqual(sum(1 for s in samples if contains_signature(s)), 1)

    def test_edge_cases(self):
        samples = [
            "",
            "   ",
            "\n\n",
            "FINAL: code ```python\nprint('🙂')\n```",
            "FINAL: markdown **bold**",
            "FINAL:\n- item1\n- item2",
            "FINAL: newline\n\nThanks,\n",  # signature positive
            "FINAL: ???",  # punct positive
            "FINAL: unicode Ω",  # negative for emoji
            "FINAL: -- zed sig",  # signature positive
        ]
        positives = [
            contains_signature(samples[6]),
            contains_special_punct(samples[7]),
            contains_signature(samples[9]),
        ]
        self.assertTrue(all(positives))


if __name__ == "__main__":
    unittest.main()
