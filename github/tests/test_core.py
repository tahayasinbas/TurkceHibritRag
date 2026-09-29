import unittest
from turkish_rag.core import hybrid_scores, retrieval_metrics, tokenize, token_windows
from turkish_rag.__main__ import evaluate
from turkish_rag.config import Settings


class FusionTests(unittest.TestCase):
    def test_paper_formula_and_weights(self):
        scores = dict(hybrid_scores({'a': 0.8, 'b': 0.5}, {'a': 2.0, 'b': 4.0}))
        self.assertAlmostEqual(scores['a'], 0.48)
        self.assertAlmostEqual(scores['b'], 0.70)
        self.assertEqual(hybrid_scores({'a': 0.8, 'b': 0.5}, {'a': 2, 'b': 4})[0][0], 'b')

    def test_all_zero_bm25_preserves_dense_order(self):
        self.assertEqual(hybrid_scores({'b': .2, 'a': .9}, {'b': 0, 'a': 0}),
                         [('a', .54), ('b', .12)])

    def test_empty_ties_and_negative_values(self):
        self.assertEqual(hybrid_scores({}, {}), [])
        scores = hybrid_scores({'b': -1, 'a': -1}, {'b': -2, 'a': -2})
        self.assertEqual(scores, [('a', -.6), ('b', -.6)])

    def test_missing_cross_channel_scores_rejected(self):
        with self.assertRaises(ValueError):
            hybrid_scores({'a': 1}, {'b': 1})

    def test_bad_scores_rejected(self):
        for val in (float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                hybrid_scores({'a': val}, {'a': 1})
        with self.assertRaises(ValueError):
            hybrid_scores({'a': 1}, {'a': 1}, 1.1)


class ChunkTests(unittest.TestCase):
    def test_overlap_and_no_redundant_tail(self):
        self.assertEqual(list(token_windows(list(range(6)), 4, 2)),
                         [(0, [0, 1, 2, 3]), (2, [2, 3, 4, 5])])
        self.assertEqual(list(token_windows(list(range(5)), 4, 2)),
                         [(0, [0, 1, 2, 3]), (2, [2, 3, 4])])
        self.assertEqual(list(token_windows([], 4, 2)), [])

    def test_invalid_overlap(self):
        with self.assertRaises(ValueError):
            list(token_windows([1], 4, 4))

    def test_turkish_casing(self):
        self.assertEqual(tokenize('İLAÇ IŞIK, Göğüs!'), ['ilaç', 'ışık', 'göğüs'])


class EvaluationTests(unittest.TestCase):
    def test_rank_two_hit_and_macro_f1(self):
        scores = retrieval_metrics(['x', 'a', 'b'], ['a', 'c'], 3)
        self.assertAlmostEqual(scores['precision_at_k'], 1 / 3)
        self.assertEqual(scores['recall_at_k'], .5)
        self.assertAlmostEqual(scores['f1_at_k'], .4)
        self.assertEqual(scores['mrr_at_k'], .5)

    def test_duplicates_do_not_inflate_score(self):
        scores = retrieval_metrics(['a', 'a', 'b'], ['a'], 3)
        self.assertAlmostEqual(scores['precision_at_k'], 1 / 3)
        self.assertEqual(scores['recall_at_k'], 1)

    def test_no_hits(self):
        self.assertTrue(all(v == 0 for v in retrieval_metrics(['x'], ['a']).values()))

    def test_unlabelled_and_failed_runs_are_not_silently_scored(self):
        with self.assertRaises(ValueError):
            retrieval_metrics(['x'], [])
        with self.assertRaises(ValueError):
            evaluate([{'error': 'RequestError'}], 5)
        with self.assertRaises(ValueError):
            evaluate([], 5)

    def test_macro_average(self):
        rows = [{'retrieved_ids': ['a'], 'relevant_ids': ['a']},
                {'retrieved_ids': ['b'], 'relevant_ids': ['a']}]
        result = evaluate(rows, 1)
        self.assertEqual(result['metrics']['mrr_at_k'], .5)
        self.assertEqual(result['n'], 2)

    def test_configuration_rejects_incompatible_candidates(self):
        with self.assertRaises(ValueError):
            Settings(candidate_k=2)


if __name__ == '__main__':
    unittest.main()
