#include <iostream>
#include <cmath>
#include <string>
#include "hmm.h"

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

// ========== Unit Tests: char_to_index ==========

void test_char_to_index() {
    std::cout << "[char_to_index] lowercase letters..." << std::endl;
    ASSERT_TRUE(char_to_index('a') == 0, "a -> 0");
    ASSERT_TRUE(char_to_index('b') == 1, "b -> 1");
    ASSERT_TRUE(char_to_index('m') == 12, "m -> 12");
    ASSERT_TRUE(char_to_index('z') == 25, "z -> 25");

    std::cout << "[char_to_index] space..." << std::endl;
    ASSERT_TRUE(char_to_index(' ') == 26, "space -> 26");

    std::cout << "[char_to_index] invalid characters..." << std::endl;
    ASSERT_THROWS(char_to_index('A'), "uppercase throws");
    ASSERT_THROWS(char_to_index('1'), "digit throws");
    ASSERT_THROWS(char_to_index('!'), "punctuation throws");
}

// ========== Unit Tests: text_to_indices ==========

void test_text_to_indices() {
    std::cout << "[text_to_indices] valid texts..." << std::endl;

    auto abc = text_to_indices("abc");
    ASSERT_TRUE(abc.size() == 3, "abc length 3");
    ASSERT_TRUE(abc[0] == 0 && abc[1] == 1 && abc[2] == 2, "abc -> {0,1,2}");

    auto ab = text_to_indices("a b");
    ASSERT_TRUE(ab.size() == 3, "a_b length 3");
    ASSERT_TRUE(ab[0] == 0 && ab[1] == 26 && ab[2] == 1, "a b -> {0,26,1}");

    auto hello = text_to_indices("hello");
    ASSERT_TRUE(hello.size() == 5, "hello length 5");
    ASSERT_TRUE(hello[0] == 7, "h -> 7");
    ASSERT_TRUE(hello[1] == 4, "e -> 4");

    std::cout << "[text_to_indices] invalid inputs..." << std::endl;
    ASSERT_THROWS(text_to_indices(""), "empty throws");
    ASSERT_THROWS(text_to_indices("Hello"), "uppercase in text throws");
}

// ========== Unit Tests: normalize ==========

void test_normalize() {
    std::cout << "[normalize] basic..." << std::endl;
    auto r = normalize({1.0, 1.0, 1.0});
    ASSERT_NEAR(r[0], 1.0 / 3.0, 1e-10, "{1,1,1} -> 1/3 each");

    auto r2 = normalize({2.0, 3.0, 5.0});
    ASSERT_NEAR(r2[0], 0.2, 1e-10, "2/(2+3+5) = 0.2");
    ASSERT_NEAR(r2[1], 0.3, 1e-10, "3/(2+3+5) = 0.3");
    ASSERT_NEAR(r2[2], 0.5, 1e-10, "5/(2+3+5) = 0.5");

    std::cout << "[normalize] sums to one..." << std::endl;
    auto r3 = normalize({7.0, 3.0, 11.0, 4.0});
    double sum = 0.0;
    for (double v : r3) sum += v;
    ASSERT_NEAR(sum, 1.0, 1e-10, "normalized sums to 1");

    std::cout << "[normalize] zero vector throws..." << std::endl;
    ASSERT_THROWS(normalize({0.0, 0.0, 0.0}), "all zeros throws");
}

// ========== Unit Tests: validate_params ==========

void test_validate_params() {
    std::cout << "[validate_params] valid..." << std::endl;
    validate_params(1);  // should not throw
    tests_passed++;
    validate_params(5);
    tests_passed++;
    validate_params(100);
    tests_passed++;

    std::cout << "[validate_params] invalid..." << std::endl;
    ASSERT_THROWS(validate_params(0), "n_states=0 throws");
    ASSERT_THROWS(validate_params(-1), "n_states=-1 throws");
}

// ========== Unit Tests: init_hmm ==========

void test_init_hmm_dimensions() {
    std::cout << "[init_hmm] dimensions..." << std::endl;
    auto hmm = init_hmm(3);
    ASSERT_TRUE(hmm.n_states == 3, "n_states == 3");
    ASSERT_TRUE(static_cast<int>(hmm.pi.size()) == 3, "pi size 3");
    ASSERT_TRUE(static_cast<int>(hmm.A.size()) == 3, "A rows 3");
    ASSERT_TRUE(static_cast<int>(hmm.A[0].size()) == 3, "A cols 3");
    ASSERT_TRUE(static_cast<int>(hmm.B.size()) == 3, "B rows 3");
    ASSERT_TRUE(static_cast<int>(hmm.B[0].size()) == N_OBSERVATIONS, "B cols 27");
}

void test_init_hmm_normalized() {
    std::cout << "[init_hmm] normalization..." << std::endl;
    auto hmm = init_hmm(4, 123);

    // pi sums to 1
    double pi_sum = 0.0;
    for (double v : hmm.pi) pi_sum += v;
    ASSERT_NEAR(pi_sum, 1.0, 1e-10, "pi sums to 1");

    // Each row of A sums to 1
    for (int i = 0; i < hmm.n_states; ++i) {
        double row_sum = 0.0;
        for (double v : hmm.A[i]) row_sum += v;
        ASSERT_NEAR(row_sum, 1.0, 1e-10, "A row " + std::to_string(i) + " sums to 1");
    }

    // Each row of B sums to 1
    for (int i = 0; i < hmm.n_states; ++i) {
        double row_sum = 0.0;
        for (double v : hmm.B[i]) row_sum += v;
        ASSERT_NEAR(row_sum, 1.0, 1e-10, "B row " + std::to_string(i) + " sums to 1");
    }
}

void test_init_hmm_values_in_range() {
    std::cout << "[init_hmm] values in [0, 1]..." << std::endl;
    auto hmm = init_hmm(5, 99);

    for (double v : hmm.pi) {
        if (v < 0.0 || v > 1.0) {
            ASSERT_TRUE(false, "pi value out of [0,1]");
            return;
        }
    }
    ASSERT_TRUE(true, "all pi values in [0,1]");

    for (const auto& row : hmm.A) {
        for (double v : row) {
            if (v < 0.0 || v > 1.0) {
                ASSERT_TRUE(false, "A value out of [0,1]");
                return;
            }
        }
    }
    ASSERT_TRUE(true, "all A values in [0,1]");

    for (const auto& row : hmm.B) {
        for (double v : row) {
            if (v < 0.0 || v > 1.0) {
                ASSERT_TRUE(false, "B value out of [0,1]");
                return;
            }
        }
    }
    ASSERT_TRUE(true, "all B values in [0,1]");
}

void test_init_hmm_deterministic() {
    std::cout << "[init_hmm] deterministic..." << std::endl;
    auto a = init_hmm(3, 42);
    auto b = init_hmm(3, 42);
    bool same = true;
    for (int i = 0; i < 3; ++i) {
        if (a.pi[i] != b.pi[i]) { same = false; break; }
    }
    ASSERT_TRUE(same, "same seed -> same HMM");

    std::cout << "[init_hmm] different seeds differ..." << std::endl;
    auto c = init_hmm(3, 1);
    auto d = init_hmm(3, 2);
    bool differ = false;
    for (int i = 0; i < 3; ++i) {
        if (c.pi[i] != d.pi[i]) { differ = true; break; }
    }
    ASSERT_TRUE(differ, "different seeds -> different HMMs");
}

// ========== Unit Tests: forward ==========

void test_forward_single_state() {
    std::cout << "[forward] single state..." << std::endl;
    // With 1 state: pi = [1], A = [[1]], B = [[b0..b26]]
    // P(O) = prod(B[0][o_t])
    auto hmm = init_hmm(1, 42);
    auto obs = text_to_indices("ab");
    auto result = forward(obs, hmm);

    double expected = std::log(hmm.B[0][0] * hmm.B[0][1]);
    ASSERT_NEAR(result.log_probability, expected, 1e-10, "single state log prob");
}

void test_forward_single_observation() {
    std::cout << "[forward] single observation..." << std::endl;
    // P(o) = sum_i(pi[i] * B[i][o])
    auto hmm = init_hmm(3, 42);
    auto obs = text_to_indices("a");
    auto result = forward(obs, hmm);

    double expected = 0.0;
    for (int i = 0; i < hmm.n_states; ++i) {
        expected += hmm.pi[i] * hmm.B[i][0];
    }
    ASSERT_NEAR(result.log_probability, std::log(expected), 1e-10,
                "single obs = sum(pi*B)");
}

void test_forward_two_observations() {
    std::cout << "[forward] two observations..." << std::endl;
    // P(o1,o2) = sum_j [sum_i(pi[i]*B[i][o1]*A[i][j]) * B[j][o2]]
    auto hmm = init_hmm(2, 42);
    auto obs = text_to_indices("ab");
    auto result = forward(obs, hmm);

    double expected = 0.0;
    for (int j = 0; j < 2; ++j) {
        double inner = 0.0;
        for (int i = 0; i < 2; ++i) {
            inner += hmm.pi[i] * hmm.B[i][0] * hmm.A[i][j];
        }
        expected += inner * hmm.B[j][1];
    }
    ASSERT_NEAR(result.log_probability, std::log(expected), 1e-10,
                "two obs forward probability");
}

void test_forward_negative_log_prob() {
    std::cout << "[forward] log probability is negative..." << std::endl;
    auto hmm = init_hmm(3, 42);
    auto obs = text_to_indices("hello");
    auto result = forward(obs, hmm);
    ASSERT_TRUE(result.log_probability < 0.0, "log prob < 0");
}

void test_forward_longer_lower_prob() {
    std::cout << "[forward] longer text -> lower log prob..." << std::endl;
    auto hmm = init_hmm(3, 42);
    auto obs_short = text_to_indices("hi");
    auto obs_long = text_to_indices("hello world");

    auto r_short = forward(obs_short, hmm);
    auto r_long = forward(obs_long, hmm);
    ASSERT_TRUE(r_long.log_probability < r_short.log_probability,
                "longer text has lower log prob");
}

void test_forward_empty_throws() {
    std::cout << "[forward] empty observations throws..." << std::endl;
    auto hmm = init_hmm(2, 42);
    ASSERT_THROWS(forward({}, hmm), "empty obs throws");
}

void test_forward_invalid_obs_throws() {
    std::cout << "[forward] invalid observation index throws..." << std::endl;
    auto hmm = init_hmm(2, 42);
    ASSERT_THROWS(forward({-1}, hmm), "negative index throws");
    ASSERT_THROWS(forward({27}, hmm), "index 27 throws");
}

void test_forward_deterministic() {
    std::cout << "[forward] deterministic results..." << std::endl;
    auto hmm = init_hmm(3, 42);
    auto obs = text_to_indices("test phrase");
    auto r1 = forward(obs, hmm);
    auto r2 = forward(obs, hmm);
    ASSERT_NEAR(r1.log_probability, r2.log_probability, 1e-15,
                "same input -> same result");
}

// ========== Integration Tests ==========

void test_full_pipeline() {
    std::cout << "[integration] full pipeline..." << std::endl;
    auto hmm = init_hmm(3, 42);

    std::vector<std::string> phrases = {
        "hello world",
        "hidden markov model",
        "speech processing",
        "the quick brown fox"
    };

    for (const auto& phrase : phrases) {
        auto obs = text_to_indices(phrase);
        auto result = forward(obs, hmm);
        ASSERT_TRUE(std::isfinite(result.log_probability),
                    "finite log prob for \"" + phrase + "\"");
        ASSERT_TRUE(result.log_probability < 0.0,
                    "negative log prob for \"" + phrase + "\"");
    }
}

void test_single_char_all_letters() {
    std::cout << "[integration] all single characters..." << std::endl;
    auto hmm = init_hmm(3, 42);

    std::string alphabet = "abcdefghijklmnopqrstuvwxyz ";
    for (char c : alphabet) {
        std::string s(1, c);
        auto obs = text_to_indices(s);
        auto result = forward(obs, hmm);
        ASSERT_TRUE(std::isfinite(result.log_probability),
                    "finite for '" + s + "'");
    }
}

void test_many_states() {
    std::cout << "[integration] many hidden states..." << std::endl;
    auto hmm = init_hmm(20, 42);
    auto obs = text_to_indices("test");
    auto result = forward(obs, hmm);
    ASSERT_TRUE(std::isfinite(result.log_probability), "finite with 20 states");
    ASSERT_TRUE(result.log_probability < 0.0, "negative with 20 states");
}

// ========== Main ==========

int main() {
    std::cout << "=== Running Tests ===" << std::endl << std::endl;

    // Unit: char_to_index
    test_char_to_index();

    // Unit: text_to_indices
    test_text_to_indices();

    // Unit: normalize
    test_normalize();

    // Unit: validate_params
    test_validate_params();

    // Unit: init_hmm
    test_init_hmm_dimensions();
    test_init_hmm_normalized();
    test_init_hmm_values_in_range();
    test_init_hmm_deterministic();

    // Unit: forward
    test_forward_single_state();
    test_forward_single_observation();
    test_forward_two_observations();
    test_forward_negative_log_prob();
    test_forward_longer_lower_prob();
    test_forward_empty_throws();
    test_forward_invalid_obs_throws();
    test_forward_deterministic();

    // Integration
    test_full_pipeline();
    test_single_char_all_letters();
    test_many_states();

    std::cout << std::endl << "=== Results ===" << std::endl;
    std::cout << "Passed: " << tests_passed << std::endl;
    std::cout << "Failed: " << tests_failed << std::endl;

    return tests_failed > 0 ? 1 : 0;
}
