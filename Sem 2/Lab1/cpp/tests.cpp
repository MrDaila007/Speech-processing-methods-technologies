#include <iostream>
#include <cmath>
#include <cassert>
#include <string>
#include <sstream>
#include "discretization.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT_TRUE(expr, msg) \
    if (!(expr)) { \
        std::cerr << "  FAIL: " << msg << " (" << #expr << ")" << std::endl; \
        tests_failed++; \
    } else { \
        tests_passed++; \
    }

#define ASSERT_NEAR(a, b, eps, msg) \
    if (std::abs((a) - (b)) > (eps)) { \
        std::cerr << "  FAIL: " << msg << " (expected " << (b) << ", got " << (a) << ")" << std::endl; \
        tests_failed++; \
    } else { \
        tests_passed++; \
    }

#define ASSERT_THROWS(expr, msg) \
    { bool threw = false; \
      try { expr; } catch (const std::exception&) { threw = true; } \
      if (!threw) { \
          std::cerr << "  FAIL: " << msg << " (no exception thrown)" << std::endl; \
          tests_failed++; \
      } else { \
          tests_passed++; \
      } \
    }

// ========== Unit Tests: quantize ==========

void test_quantize_exact_level() {
    std::cout << "[quantize] exact level value..." << std::endl;
    // 8-bit: step = 19980 / 255 = 78.3529...
    double step = compute_step(8);
    // Value exactly at level 0 (FREQ_MIN)
    ASSERT_NEAR(quantize(20.0, step), 20.0, 0.001, "quantize(20) == 20");
    // Value exactly at level 255 (FREQ_MAX)
    ASSERT_NEAR(quantize(20000.0, step), 20000.0, 0.001, "quantize(20000) == 20000");
}

void test_quantize_rounds_to_nearest() {
    std::cout << "[quantize] rounds to nearest level..." << std::endl;
    double step = compute_step(8); // ~78.35
    // Value 59.0 is between level 0 (20.0) and level 1 (98.35)
    // 59.0 - 20.0 = 39.0, 39.0/78.35 = 0.498 -> rounds to 0 -> 20.0
    ASSERT_NEAR(quantize(59.0, step), 20.0, 0.01, "59 rounds down to 20");
    // Value 60.0: (60-20)/78.35 = 0.51 -> rounds to 1 -> 98.35
    ASSERT_NEAR(quantize(60.0, step), 20.0 + step, 0.01, "60 rounds up to next level");
}

void test_quantize_midrange() {
    std::cout << "[quantize] midrange value..." << std::endl;
    double step = compute_step(16);
    double mid = 10010.0;
    double result = quantize(mid, step);
    // Result should be within one step of input
    ASSERT_TRUE(std::abs(result - mid) <= step, "quantized within one step of input");
    // Result should be on a valid level: (result - FREQ_MIN) / step is integer
    double level_f = (result - FREQ_MIN) / step;
    ASSERT_NEAR(level_f, std::round(level_f), 1e-6, "result is on exact level");
}

// ========== Unit Tests: compute_step ==========

void test_compute_step() {
    std::cout << "[compute_step] various bitrates..." << std::endl;
    // 1-bit: 2 levels (0,1) -> 1 interval -> step = 19980
    ASSERT_NEAR(compute_step(1), 19980.0, 0.001, "1-bit step = 19980");
    // 8-bit: 255 intervals -> step = 19980/255
    ASSERT_NEAR(compute_step(8), 19980.0 / 255.0, 0.001, "8-bit step");
    // 16-bit: 65535 intervals
    ASSERT_NEAR(compute_step(16), 19980.0 / 65535.0, 1e-6, "16-bit step");
}

// ========== Unit Tests: generate_signal ==========

void test_generate_signal_count() {
    std::cout << "[generate_signal] correct count..." << std::endl;
    auto sig = generate_signal(1000);
    ASSERT_TRUE(sig.size() == 1000, "generated 1000 values");
}

void test_generate_signal_range() {
    std::cout << "[generate_signal] values in range..." << std::endl;
    auto sig = generate_signal(10000);
    for (size_t i = 0; i < sig.size(); ++i) {
        if (sig[i] < FREQ_MIN || sig[i] > FREQ_MAX) {
            ASSERT_TRUE(false, "value out of range at index " + std::to_string(i));
            return;
        }
    }
    ASSERT_TRUE(true, "all values in [20, 20000]");
}

void test_generate_signal_deterministic() {
    std::cout << "[generate_signal] deterministic with same seed..." << std::endl;
    auto a = generate_signal(100, 123);
    auto b = generate_signal(100, 123);
    bool same = true;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) { same = false; break; }
    }
    ASSERT_TRUE(same, "same seed produces same signal");
}

void test_generate_signal_different_seeds() {
    std::cout << "[generate_signal] different seeds differ..." << std::endl;
    auto a = generate_signal(100, 1);
    auto b = generate_signal(100, 2);
    bool differ = false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) { differ = true; break; }
    }
    ASSERT_TRUE(differ, "different seeds produce different signals");
}

// ========== Unit Tests: validate_params ==========

void test_validate_params_valid() {
    std::cout << "[validate_params] valid inputs..." << std::endl;
    auto p = validate_params(1, 8, 48);
    ASSERT_TRUE(p.n_seconds == 1, "n_seconds == 1");
    ASSERT_TRUE(p.bitrate == 8, "bitrate == 8");
    ASSERT_TRUE(p.sample_rate_khz == 48, "sample_rate == 48");
}

void test_validate_params_all_rates() {
    std::cout << "[validate_params] all valid sample rates..." << std::endl;
    const int rates[] = {48, 96, 128, 176, 256};
    for (int r : rates) {
        auto p = validate_params(1, 8, r);
        ASSERT_TRUE(p.sample_rate_khz == r, "rate " + std::to_string(r) + " accepted");
    }
}

void test_validate_params_invalid_n() {
    std::cout << "[validate_params] rejects n <= 0..." << std::endl;
    ASSERT_THROWS(validate_params(0, 8, 48), "n=0 throws");
    ASSERT_THROWS(validate_params(-1, 8, 48), "n=-1 throws");
}

void test_validate_params_invalid_bitrate() {
    std::cout << "[validate_params] rejects bad bitrate..." << std::endl;
    ASSERT_THROWS(validate_params(1, 0, 48), "bitrate=0 throws");
    ASSERT_THROWS(validate_params(1, 25, 48), "bitrate=25 throws");
    ASSERT_THROWS(validate_params(1, -1, 48), "bitrate=-1 throws");
}

void test_validate_params_invalid_rate() {
    std::cout << "[validate_params] rejects bad sample rate..." << std::endl;
    ASSERT_THROWS(validate_params(1, 8, 44), "rate=44 throws");
    ASSERT_THROWS(validate_params(1, 8, 100), "rate=100 throws");
}

// ========== Unit Tests: is_valid_sample_rate ==========

void test_is_valid_sample_rate() {
    std::cout << "[is_valid_sample_rate] checks..." << std::endl;
    ASSERT_TRUE(is_valid_sample_rate(48), "48 valid");
    ASSERT_TRUE(is_valid_sample_rate(96), "96 valid");
    ASSERT_TRUE(is_valid_sample_rate(128), "128 valid");
    ASSERT_TRUE(is_valid_sample_rate(176), "176 valid");
    ASSERT_TRUE(is_valid_sample_rate(256), "256 valid");
    ASSERT_TRUE(!is_valid_sample_rate(44), "44 invalid");
    ASSERT_TRUE(!is_valid_sample_rate(0), "0 invalid");
    ASSERT_TRUE(!is_valid_sample_rate(192), "192 invalid");
}

// ========== Integration Tests: discretize ==========

void test_discretize_output_count() {
    std::cout << "[discretize] correct output sample count..." << std::endl;
    // 100 input values, 1 second, 48 KHz -> 48000 samples
    // Use small input to keep test fast
    std::vector<double> signal(100000, 10000.0);
    auto output = discretize(signal, 1, 8, 48);
    ASSERT_TRUE(output.size() == 48000, "48000 output samples for 48KHz * 1s");
}

void test_discretize_output_count_other_rates() {
    std::cout << "[discretize] output count for various rates..." << std::endl;
    std::vector<double> signal(500000, 5000.0);

    auto o96 = discretize(signal, 1, 8, 96);
    ASSERT_TRUE(o96.size() == 96000, "96000 samples for 96KHz");

    auto o256 = discretize(signal, 1, 8, 256);
    ASSERT_TRUE(o256.size() == 256000, "256000 samples for 256KHz");
}

void test_discretize_constant_signal() {
    std::cout << "[discretize] constant signal preserves value..." << std::endl;
    double step = compute_step(8);
    double exact_level = FREQ_MIN + 100 * step; // exact level 100
    std::vector<double> signal(100000, exact_level);

    auto output = discretize(signal, 1, 8, 48);
    for (size_t i = 0; i < output.size(); ++i) {
        if (std::abs(output[i] - exact_level) > 0.001) {
            ASSERT_TRUE(false, "constant signal not preserved at index " + std::to_string(i));
            return;
        }
    }
    ASSERT_TRUE(true, "all samples match constant input");
}

void test_discretize_values_in_range() {
    std::cout << "[discretize] output values in [FREQ_MIN, FREQ_MAX]..." << std::endl;
    auto signal = generate_signal(100000);
    auto output = discretize(signal, 1, 8, 48);

    for (size_t i = 0; i < output.size(); ++i) {
        if (output[i] < FREQ_MIN - 0.001 || output[i] > FREQ_MAX + 0.001) {
            ASSERT_TRUE(false, "output out of range at " + std::to_string(i));
            return;
        }
    }
    ASSERT_TRUE(true, "all output values in valid range");
}

void test_discretize_values_on_levels() {
    std::cout << "[discretize] output values are on quantization levels..." << std::endl;
    auto signal = generate_signal(100000);
    auto output = discretize(signal, 1, 8, 48);
    double step = compute_step(8);

    for (size_t i = 0; i < std::min(output.size(), (size_t)1000); ++i) {
        double level_f = (output[i] - FREQ_MIN) / step;
        double rounded = std::round(level_f);
        if (std::abs(level_f - rounded) > 1e-6) {
            ASSERT_TRUE(false, "not on level at index " + std::to_string(i));
            return;
        }
    }
    ASSERT_TRUE(true, "all checked samples on valid levels");
}

void test_discretize_averaging() {
    std::cout << "[discretize] averaging works correctly..." << std::endl;
    // Create signal where we know the average of each bin
    // 4 input values, 1 second, but we need sample_rate to be valid
    // Use: 2 values per sample -> 2 samples from 4 values
    // But min sample_rate is 48KHz, so use 48000 samples from 96000 values
    std::vector<double> signal(96000);
    // Fill pairs: [100, 200, 300, 400, ...]
    // Each pair averages: 150, 350, ...
    for (size_t i = 0; i < signal.size(); i += 2) {
        signal[i] = 100.0;
        signal[i + 1] = 200.0;
    }
    // Average of each bin of 2 values = 150.0
    auto output = discretize(signal, 1, 16, 48);
    double step = compute_step(16);
    double expected = quantize(150.0, step);

    // Check first several samples
    bool all_correct = true;
    for (size_t i = 0; i < 100; ++i) {
        if (std::abs(output[i] - expected) > 0.01) {
            all_correct = false;
            std::cerr << "  at " << i << ": expected " << expected << ", got " << output[i] << std::endl;
            break;
        }
    }
    ASSERT_TRUE(all_correct, "bins average correctly to quantized 150");
}

void test_discretize_multi_second() {
    std::cout << "[discretize] multi-second signal..." << std::endl;
    int n = 2;
    std::vector<double> signal(200000, 10000.0);
    auto output = discretize(signal, n, 8, 48);
    ASSERT_TRUE(output.size() == 96000, "2 seconds * 48000 = 96000 samples");
}

// ========== Edge case tests ==========

void test_quantize_at_boundaries() {
    std::cout << "[quantize] boundary values..." << std::endl;
    double step = compute_step(8);
    // Just above FREQ_MIN
    double result_low = quantize(20.1, step);
    ASSERT_TRUE(result_low >= FREQ_MIN, "low boundary >= FREQ_MIN");

    // Just below FREQ_MAX
    double result_high = quantize(19999.9, step);
    ASSERT_TRUE(result_high <= FREQ_MAX + 0.001, "high boundary <= FREQ_MAX");
}

void test_discretize_1bit() {
    std::cout << "[discretize] 1-bit quantization (2 levels)..." << std::endl;
    // 1-bit: only 2 levels: 20 and 20000
    auto signal = generate_signal(100000);
    auto output = discretize(signal, 1, 1, 48);

    for (size_t i = 0; i < output.size(); ++i) {
        bool valid = (std::abs(output[i] - FREQ_MIN) < 0.01) ||
                     (std::abs(output[i] - FREQ_MAX) < 0.01);
        if (!valid) {
            ASSERT_TRUE(false, "1-bit value not 20 or 20000 at index " + std::to_string(i));
            return;
        }
    }
    ASSERT_TRUE(true, "all 1-bit values are 20 or 20000");
}

void test_discretize_high_bitrate() {
    std::cout << "[discretize] high bitrate precision..." << std::endl;
    double step_8 = compute_step(8);
    double step_16 = compute_step(16);
    ASSERT_TRUE(step_16 < step_8, "16-bit step < 8-bit step");

    // Higher bitrate should produce closer approximation
    std::vector<double> signal(100000, 12345.6789);
    auto out8 = discretize(signal, 1, 8, 48);
    auto out16 = discretize(signal, 1, 16, 48);

    double err8 = std::abs(out8[0] - 12345.6789);
    double err16 = std::abs(out16[0] - 12345.6789);
    ASSERT_TRUE(err16 <= err8, "16-bit error <= 8-bit error");
}

// ========== Main ==========

int main() {
    std::cout << "=== Running Tests ===" << std::endl << std::endl;

    // Unit: quantize
    test_quantize_exact_level();
    test_quantize_rounds_to_nearest();
    test_quantize_midrange();
    test_quantize_at_boundaries();

    // Unit: compute_step
    test_compute_step();

    // Unit: generate_signal
    test_generate_signal_count();
    test_generate_signal_range();
    test_generate_signal_deterministic();
    test_generate_signal_different_seeds();

    // Unit: validate_params
    test_validate_params_valid();
    test_validate_params_all_rates();
    test_validate_params_invalid_n();
    test_validate_params_invalid_bitrate();
    test_validate_params_invalid_rate();

    // Unit: is_valid_sample_rate
    test_is_valid_sample_rate();

    // Integration: discretize
    test_discretize_output_count();
    test_discretize_output_count_other_rates();
    test_discretize_constant_signal();
    test_discretize_values_in_range();
    test_discretize_values_on_levels();
    test_discretize_averaging();
    test_discretize_multi_second();

    // Edge cases
    test_discretize_1bit();
    test_discretize_high_bitrate();

    std::cout << std::endl << "=== Results ===" << std::endl;
    std::cout << "Passed: " << tests_passed << std::endl;
    std::cout << "Failed: " << tests_failed << std::endl;

    return tests_failed > 0 ? 1 : 0;
}
