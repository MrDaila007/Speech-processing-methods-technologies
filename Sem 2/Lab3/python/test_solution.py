import unittest
import math
from solution import (
    N_OBSERVATIONS,
    validate_params, char_to_index, text_to_indices,
    normalize, init_hmm, forward,
)


class TestCharToIndex(unittest.TestCase):
    def test_a(self):
        self.assertEqual(char_to_index('a'), 0)

    def test_z(self):
        self.assertEqual(char_to_index('z'), 25)

    def test_m(self):
        self.assertEqual(char_to_index('m'), 12)

    def test_space(self):
        self.assertEqual(char_to_index(' '), 26)

    def test_uppercase_throws(self):
        with self.assertRaises(ValueError):
            char_to_index('A')

    def test_digit_throws(self):
        with self.assertRaises(ValueError):
            char_to_index('1')

    def test_punctuation_throws(self):
        with self.assertRaises(ValueError):
            char_to_index('!')


class TestTextToIndices(unittest.TestCase):
    def test_abc(self):
        self.assertEqual(text_to_indices("abc"), [0, 1, 2])

    def test_with_space(self):
        self.assertEqual(text_to_indices("a b"), [0, 26, 1])

    def test_hello(self):
        result = text_to_indices("hello")
        self.assertEqual(len(result), 5)
        self.assertEqual(result[0], 7)  # h

    def test_empty_throws(self):
        with self.assertRaises(ValueError):
            text_to_indices("")

    def test_uppercase_throws(self):
        with self.assertRaises(ValueError):
            text_to_indices("Hello")


class TestNormalize(unittest.TestCase):
    def test_uniform(self):
        result = normalize([1.0, 1.0, 1.0])
        for v in result:
            self.assertAlmostEqual(v, 1.0 / 3.0, places=10)

    def test_weighted(self):
        result = normalize([2.0, 3.0, 5.0])
        self.assertAlmostEqual(result[0], 0.2, places=10)
        self.assertAlmostEqual(result[1], 0.3, places=10)
        self.assertAlmostEqual(result[2], 0.5, places=10)

    def test_sums_to_one(self):
        result = normalize([7.0, 3.0, 11.0, 4.0])
        self.assertAlmostEqual(sum(result), 1.0, places=10)

    def test_zeros_throws(self):
        with self.assertRaises(ValueError):
            normalize([0.0, 0.0, 0.0])


class TestValidateParams(unittest.TestCase):
    def test_valid(self):
        validate_params(1)
        validate_params(5)

    def test_zero_throws(self):
        with self.assertRaises(ValueError):
            validate_params(0)

    def test_negative_throws(self):
        with self.assertRaises(ValueError):
            validate_params(-1)


class TestInitHmm(unittest.TestCase):
    def test_dimensions(self):
        hmm = init_hmm(3)
        self.assertEqual(hmm['n_states'], 3)
        self.assertEqual(len(hmm['pi']), 3)
        self.assertEqual(len(hmm['A']), 3)
        self.assertEqual(len(hmm['A'][0]), 3)
        self.assertEqual(len(hmm['B']), 3)
        self.assertEqual(len(hmm['B'][0]), N_OBSERVATIONS)

    def test_pi_normalized(self):
        hmm = init_hmm(4, seed=123)
        self.assertAlmostEqual(sum(hmm['pi']), 1.0, places=10)

    def test_A_rows_normalized(self):
        hmm = init_hmm(4, seed=123)
        for row in hmm['A']:
            self.assertAlmostEqual(sum(row), 1.0, places=10)

    def test_B_rows_normalized(self):
        hmm = init_hmm(4, seed=123)
        for row in hmm['B']:
            self.assertAlmostEqual(sum(row), 1.0, places=10)

    def test_values_in_range(self):
        hmm = init_hmm(5, seed=99)
        for v in hmm['pi']:
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)
        for row in hmm['A']:
            for v in row:
                self.assertGreaterEqual(v, 0.0)
                self.assertLessEqual(v, 1.0)

    def test_deterministic(self):
        a = init_hmm(3, seed=42)
        b = init_hmm(3, seed=42)
        self.assertEqual(a['pi'], b['pi'])

    def test_different_seeds(self):
        a = init_hmm(3, seed=1)
        b = init_hmm(3, seed=2)
        self.assertNotEqual(a['pi'], b['pi'])


class TestForward(unittest.TestCase):
    def test_single_state(self):
        hmm = init_hmm(1, seed=42)
        obs = text_to_indices("ab")
        result = forward(obs, hmm)
        expected = math.log(hmm['B'][0][0] * hmm['B'][0][1])
        self.assertAlmostEqual(result['log_probability'], expected, places=10)

    def test_single_observation(self):
        hmm = init_hmm(3, seed=42)
        obs = text_to_indices("a")
        result = forward(obs, hmm)
        expected = sum(hmm['pi'][i] * hmm['B'][i][0] for i in range(3))
        self.assertAlmostEqual(result['log_probability'], math.log(expected), places=10)

    def test_two_observations(self):
        hmm = init_hmm(2, seed=42)
        obs = text_to_indices("ab")
        result = forward(obs, hmm)
        expected = 0.0
        for j in range(2):
            inner = sum(hmm['pi'][i] * hmm['B'][i][0] * hmm['A'][i][j]
                        for i in range(2))
            expected += inner * hmm['B'][j][1]
        self.assertAlmostEqual(result['log_probability'], math.log(expected), places=10)

    def test_negative_log_prob(self):
        hmm = init_hmm(3, seed=42)
        obs = text_to_indices("hello")
        result = forward(obs, hmm)
        self.assertLess(result['log_probability'], 0.0)

    def test_longer_lower_prob(self):
        hmm = init_hmm(3, seed=42)
        r_short = forward(text_to_indices("hi"), hmm)
        r_long = forward(text_to_indices("hello world"), hmm)
        self.assertLess(r_long['log_probability'], r_short['log_probability'])

    def test_empty_throws(self):
        hmm = init_hmm(2, seed=42)
        with self.assertRaises(ValueError):
            forward([], hmm)

    def test_invalid_index_throws(self):
        hmm = init_hmm(2, seed=42)
        with self.assertRaises(ValueError):
            forward([-1], hmm)
        with self.assertRaises(ValueError):
            forward([27], hmm)

    def test_deterministic(self):
        hmm = init_hmm(3, seed=42)
        obs = text_to_indices("test phrase")
        r1 = forward(obs, hmm)
        r2 = forward(obs, hmm)
        self.assertEqual(r1['log_probability'], r2['log_probability'])


class TestIntegration(unittest.TestCase):
    def test_full_pipeline(self):
        hmm = init_hmm(3, seed=42)
        phrases = ["hello world", "hidden markov model",
                    "speech processing", "the quick brown fox"]
        for phrase in phrases:
            obs = text_to_indices(phrase)
            result = forward(obs, hmm)
            self.assertTrue(math.isfinite(result['log_probability']))
            self.assertLess(result['log_probability'], 0.0)

    def test_many_states(self):
        hmm = init_hmm(20, seed=42)
        obs = text_to_indices("test")
        result = forward(obs, hmm)
        self.assertTrue(math.isfinite(result['log_probability']))


if __name__ == "__main__":
    unittest.main()
