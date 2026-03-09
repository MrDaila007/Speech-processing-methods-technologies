import unittest
import math
from solution import (
    FREQ_MIN, FREQ_MAX, VALID_RATES,
    validate_params, generate_signal, compute_step, quantize, discretize,
)


class TestComputeStep(unittest.TestCase):
    def test_1bit(self):
        self.assertAlmostEqual(compute_step(1), 19980.0, places=3)

    def test_8bit(self):
        self.assertAlmostEqual(compute_step(8), 19980.0 / 255, places=3)

    def test_16bit(self):
        self.assertAlmostEqual(compute_step(16), 19980.0 / 65535, places=6)


class TestQuantize(unittest.TestCase):
    def test_exact_min(self):
        step = compute_step(8)
        self.assertAlmostEqual(quantize(20.0, step), 20.0, places=3)

    def test_exact_max(self):
        step = compute_step(8)
        self.assertAlmostEqual(quantize(20000.0, step), 20000.0, places=3)

    def test_rounds_down(self):
        step = compute_step(8)  # ~78.35
        result = quantize(59.0, step)
        self.assertAlmostEqual(result, 20.0, places=2)

    def test_rounds_up(self):
        step = compute_step(8)
        result = quantize(60.0, step)
        self.assertAlmostEqual(result, 20.0 + step, places=2)

    def test_midrange_within_step(self):
        step = compute_step(16)
        mid = 10010.0
        result = quantize(mid, step)
        self.assertLessEqual(abs(result - mid), step)

    def test_result_on_level(self):
        step = compute_step(16)
        result = quantize(10010.0, step)
        level_f = (result - FREQ_MIN) / step
        self.assertAlmostEqual(level_f, round(level_f), places=6)

    def test_boundary_low(self):
        step = compute_step(8)
        result = quantize(20.1, step)
        self.assertGreaterEqual(result, FREQ_MIN)

    def test_boundary_high(self):
        step = compute_step(8)
        result = quantize(19999.9, step)
        self.assertLessEqual(result, FREQ_MAX + 0.001)


class TestGenerateSignal(unittest.TestCase):
    def test_count(self):
        sig = generate_signal(1000)
        self.assertEqual(len(sig), 1000)

    def test_range(self):
        sig = generate_signal(10000)
        for v in sig:
            self.assertGreaterEqual(v, FREQ_MIN)
            self.assertLessEqual(v, FREQ_MAX)

    def test_deterministic(self):
        a = generate_signal(100, seed=123)
        b = generate_signal(100, seed=123)
        self.assertEqual(a, b)

    def test_different_seeds(self):
        a = generate_signal(100, seed=1)
        b = generate_signal(100, seed=2)
        self.assertNotEqual(a, b)


class TestValidateParams(unittest.TestCase):
    def test_valid(self):
        n, br, sr = validate_params(1, 8, 48)
        self.assertEqual((n, br, sr), (1, 8, 48))

    def test_all_valid_rates(self):
        for r in VALID_RATES:
            _, _, sr = validate_params(1, 8, r)
            self.assertEqual(sr, r)

    def test_invalid_n_zero(self):
        with self.assertRaises(ValueError):
            validate_params(0, 8, 48)

    def test_invalid_n_negative(self):
        with self.assertRaises(ValueError):
            validate_params(-1, 8, 48)

    def test_invalid_bitrate_zero(self):
        with self.assertRaises(ValueError):
            validate_params(1, 0, 48)

    def test_invalid_bitrate_high(self):
        with self.assertRaises(ValueError):
            validate_params(1, 25, 48)

    def test_invalid_rate(self):
        with self.assertRaises(ValueError):
            validate_params(1, 8, 44)


class TestDiscretize(unittest.TestCase):
    def test_output_count_48khz(self):
        signal = [10000.0] * 100000
        output = discretize(signal, 1, 8, 48)
        self.assertEqual(len(output), 48000)

    def test_output_count_96khz(self):
        signal = [5000.0] * 500000
        output = discretize(signal, 1, 8, 96)
        self.assertEqual(len(output), 96000)

    def test_output_count_256khz(self):
        signal = [5000.0] * 500000
        output = discretize(signal, 1, 8, 256)
        self.assertEqual(len(output), 256000)

    def test_constant_signal(self):
        step = compute_step(8)
        exact_level = FREQ_MIN + 100 * step
        signal = [exact_level] * 100000
        output = discretize(signal, 1, 8, 48)
        for v in output:
            self.assertAlmostEqual(v, exact_level, places=3)

    def test_values_in_range(self):
        signal = generate_signal(100000)
        output = discretize(signal, 1, 8, 48)
        for v in output:
            self.assertGreaterEqual(v, FREQ_MIN - 0.001)
            self.assertLessEqual(v, FREQ_MAX + 0.001)

    def test_values_on_levels(self):
        signal = generate_signal(100000)
        output = discretize(signal, 1, 8, 48)
        step = compute_step(8)
        for v in output[:1000]:
            level_f = (v - FREQ_MIN) / step
            self.assertAlmostEqual(level_f, round(level_f), places=6)

    def test_averaging(self):
        signal = []
        for _ in range(48000):
            signal.extend([100.0, 200.0])
        # Each bin of 2 values averages to 150.0
        output = discretize(signal, 1, 16, 48)
        step = compute_step(16)
        expected = quantize(150.0, step)
        for v in output[:100]:
            self.assertAlmostEqual(v, expected, places=2)

    def test_multi_second(self):
        signal = [10000.0] * 200000
        output = discretize(signal, 2, 8, 48)
        self.assertEqual(len(output), 96000)

    def test_1bit_two_levels(self):
        signal = generate_signal(100000)
        output = discretize(signal, 1, 1, 48)
        for v in output:
            self.assertTrue(
                abs(v - FREQ_MIN) < 0.01 or abs(v - FREQ_MAX) < 0.01,
                f"1-bit value {v} not 20 or 20000",
            )

    def test_higher_bitrate_less_error(self):
        signal = [12345.6789] * 100000
        out8 = discretize(signal, 1, 8, 48)
        out16 = discretize(signal, 1, 16, 48)
        err8 = abs(out8[0] - 12345.6789)
        err16 = abs(out16[0] - 12345.6789)
        self.assertLessEqual(err16, err8)


if __name__ == "__main__":
    unittest.main()
