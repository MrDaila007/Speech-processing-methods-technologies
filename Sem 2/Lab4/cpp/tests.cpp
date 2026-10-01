#include <iostream>
#include <cmath>
#include <cassert>
#include <string>
#include <numeric>
#include "mlp.h"

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

// ========== Unit Tests: load_iris ==========

void test_load_iris() {
    std::cout << "[load_iris] shape and labels..." << std::endl;
    auto data = load_iris();
    ASSERT_TRUE(data.size() == 150, "150 samples");

    int count0 = 0, count1 = 0, count2 = 0;
    for (const auto& s : data) {
        if (s.label == 0) count0++;
        else if (s.label == 1) count1++;
        else if (s.label == 2) count2++;
    }
    ASSERT_TRUE(count0 == 50, "50 setosa");
    ASSERT_TRUE(count1 == 50, "50 versicolor");
    ASSERT_TRUE(count2 == 50, "50 virginica");

    // Check first sample
    ASSERT_NEAR(data[0].features[0], 5.1, 1e-6, "first sample f0=5.1");
    ASSERT_NEAR(data[0].features[1], 3.5, 1e-6, "first sample f1=3.5");
    ASSERT_NEAR(data[0].features[2], 1.4, 1e-6, "first sample f2=1.4");
    ASSERT_NEAR(data[0].features[3], 0.2, 1e-6, "first sample f3=0.2");
    ASSERT_TRUE(data[0].label == 0, "first sample label=0");

    // Check a versicolor sample (index 50)
    ASSERT_NEAR(data[50].features[0], 7.0, 1e-6, "sample 50 f0=7.0");
    ASSERT_TRUE(data[50].label == 1, "sample 50 label=1");

    // Check a virginica sample (index 100)
    ASSERT_NEAR(data[100].features[0], 6.3, 1e-6, "sample 100 f0=6.3");
    ASSERT_TRUE(data[100].label == 2, "sample 100 label=2");

    // All labels valid
    bool all_valid = true;
    for (const auto& s : data) {
        if (s.label < 0 || s.label > 2) { all_valid = false; break; }
    }
    ASSERT_TRUE(all_valid, "all labels in {0,1,2}");
}

// ========== Unit Tests: split_data ==========

void test_split_data() {
    std::cout << "[split_data] sizes and determinism..." << std::endl;
    auto data = load_iris();
    std::vector<std::vector<double>> X(data.size());
    std::vector<int> y(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        X[i] = {data[i].features[0], data[i].features[1],
                 data[i].features[2], data[i].features[3]};
        y[i] = data[i].label;
    }

    auto s = split_data(X, y, 0.8, 42);
    ASSERT_TRUE(s.X_train.size() == 120, "120 train samples");
    ASSERT_TRUE(s.X_test.size() == 30, "30 test samples");
    ASSERT_TRUE(s.y_train.size() == 120, "120 train labels");
    ASSERT_TRUE(s.y_test.size() == 30, "30 test labels");

    // Deterministic: same seed -> same split
    auto s2 = split_data(X, y, 0.8, 42);
    bool same = true;
    for (size_t i = 0; i < s.y_train.size(); ++i) {
        if (s.y_train[i] != s2.y_train[i]) { same = false; break; }
    }
    ASSERT_TRUE(same, "same seed -> same split");

    // Different seed -> different split
    auto s3 = split_data(X, y, 0.8, 99);
    bool differ = false;
    for (size_t i = 0; i < s.y_train.size(); ++i) {
        if (s.y_train[i] != s3.y_train[i]) { differ = true; break; }
    }
    ASSERT_TRUE(differ, "different seed -> different split");

    // Error on empty
    ASSERT_THROWS(split_data({}, {}, 0.8, 42), "empty data throws");
    ASSERT_THROWS(split_data(X, y, 0.0, 42), "ratio 0 throws");
    ASSERT_THROWS(split_data(X, y, 1.0, 42), "ratio 1 throws");
}

// ========== Unit Tests: normalize ==========

void test_normalize() {
    std::cout << "[normalize] mean~0, std~1..." << std::endl;
    auto data = load_iris();
    std::vector<std::vector<double>> X(data.size());
    std::vector<int> y(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        X[i] = {data[i].features[0], data[i].features[1],
                 data[i].features[2], data[i].features[3]};
        y[i] = data[i].label;
    }
    auto s = split_data(X, y, 0.8, 42);
    auto norm = normalize(s.X_train, s.X_test);

    int nf = 4;
    int n = static_cast<int>(norm.X_train.size());

    for (int j = 0; j < nf; ++j) {
        double mean = 0.0;
        for (int i = 0; i < n; ++i) mean += norm.X_train[i][j];
        mean /= n;
        ASSERT_NEAR(mean, 0.0, 1e-10, "train feature " + std::to_string(j) + " mean~0");
    }

    for (int j = 0; j < nf; ++j) {
        double mean = 0.0;
        for (int i = 0; i < n; ++i) mean += norm.X_train[i][j];
        mean /= n;
        double var = 0.0;
        for (int i = 0; i < n; ++i) {
            double d = norm.X_train[i][j] - mean;
            var += d * d;
        }
        double std_dev = std::sqrt(var / n);
        ASSERT_NEAR(std_dev, 1.0, 1e-10, "train feature " + std::to_string(j) + " std~1");
    }

    // Test set transformed
    ASSERT_TRUE(norm.X_test.size() == s.X_test.size(), "test size preserved");

    // Error on empty
    std::vector<std::vector<double>> empty;
    ASSERT_THROWS(normalize(empty, empty), "empty train throws");
}

// ========== Unit Tests: Activation Functions ==========

void test_relu() {
    std::cout << "[relu] key values..." << std::endl;
    ASSERT_NEAR(relu(0.0), 0.0, 1e-12, "relu(0)=0");
    ASSERT_NEAR(relu(1.0), 1.0, 1e-12, "relu(1)=1");
    ASSERT_NEAR(relu(-1.0), 0.0, 1e-12, "relu(-1)=0");
    ASSERT_NEAR(relu(5.5), 5.5, 1e-12, "relu(5.5)=5.5");

    ASSERT_NEAR(relu_deriv(1.0), 1.0, 1e-12, "relu_deriv(1)=1");
    ASSERT_NEAR(relu_deriv(-1.0), 0.0, 1e-12, "relu_deriv(-1)=0");
    ASSERT_NEAR(relu_deriv(0.0), 0.0, 1e-12, "relu_deriv(0)=0");
}

void test_sigmoid() {
    std::cout << "[sigmoid] key values..." << std::endl;
    ASSERT_NEAR(sigmoid(0.0), 0.5, 1e-12, "sigmoid(0)=0.5");
    ASSERT_TRUE(sigmoid(10.0) > 0.999, "sigmoid(10)>0.999");
    ASSERT_TRUE(sigmoid(-10.0) < 0.001, "sigmoid(-10)<0.001");

    ASSERT_NEAR(sigmoid_deriv(0.0), 0.25, 1e-12, "sigmoid_deriv(0)=0.25");
    ASSERT_TRUE(sigmoid_deriv(10.0) < 0.001, "sigmoid_deriv(10)<0.001");
}

void test_bipolar_sigmoid() {
    std::cout << "[bipolar_sigmoid] key values..." << std::endl;
    ASSERT_NEAR(bipolar_sigmoid(0.0), 0.0, 1e-12, "bipolar(0)=0");
    ASSERT_TRUE(bipolar_sigmoid(10.0) > 0.99, "bipolar(10)>0.99");
    ASSERT_TRUE(bipolar_sigmoid(-10.0) < -0.99, "bipolar(-10)<-0.99");

    ASSERT_NEAR(bipolar_sigmoid_deriv(0.0), 0.5, 1e-12, "bipolar_deriv(0)=0.5");
}

void test_gelu() {
    std::cout << "[gelu] key values..." << std::endl;
    ASSERT_NEAR(gelu(0.0), 0.0, 1e-12, "gelu(0)=0");
    ASSERT_TRUE(gelu(3.0) > 2.99, "gelu(3)~3");
    ASSERT_TRUE(gelu(-3.0) < 0.01 && gelu(-3.0) > -0.02, "gelu(-3)~0");

    // gelu_deriv(0) = 0.5*(1+tanh(0)) + 0 = 0.5
    ASSERT_NEAR(gelu_deriv(0.0), 0.5, 1e-6, "gelu_deriv(0)=0.5");
    ASSERT_TRUE(gelu_deriv(3.0) > 0.99, "gelu_deriv(3)~1");
}

// ========== Unit Tests: softmax ==========

void test_softmax() {
    std::cout << "[softmax] sum=1, stability..." << std::endl;

    auto s1 = softmax({1.0, 2.0, 3.0});
    double sum1 = 0.0;
    for (double v : s1) sum1 += v;
    ASSERT_NEAR(sum1, 1.0, 1e-12, "softmax sums to 1");

    ASSERT_TRUE(s1[2] > s1[1] && s1[1] > s1[0], "softmax preserves order");

    // All equal -> uniform
    auto s2 = softmax({1.0, 1.0, 1.0});
    ASSERT_NEAR(s2[0], 1.0/3.0, 1e-12, "equal inputs -> 1/3 each");

    // Numerical stability with large values
    auto s3 = softmax({1000.0, 1001.0, 1002.0});
    double sum3 = 0.0;
    for (double v : s3) sum3 += v;
    ASSERT_NEAR(sum3, 1.0, 1e-10, "softmax stable with large values");

    // All values in [0,1]
    bool all_valid = true;
    for (double v : s3) {
        if (v < 0.0 || v > 1.0) { all_valid = false; break; }
    }
    ASSERT_TRUE(all_valid, "softmax values in [0,1]");

    // Empty throws
    ASSERT_THROWS(softmax({}), "softmax empty throws");
}

// ========== Unit Tests: validate_params ==========

void test_validate_params() {
    std::cout << "[validate_params] valid and invalid..." << std::endl;
    // Valid: should not throw
    validate_params(4, 16, 3);
    tests_passed++;
    validate_params(1, 1, 1);
    tests_passed++;

    // Invalid
    ASSERT_THROWS(validate_params(0, 16, 3), "n_in=0 throws");
    ASSERT_THROWS(validate_params(4, 0, 3), "n_hidden=0 throws");
    ASSERT_THROWS(validate_params(4, 16, 0), "n_out=0 throws");
    ASSERT_THROWS(validate_params(-1, 16, 3), "n_in=-1 throws");
    ASSERT_THROWS(validate_params(4, -5, 3), "n_hidden=-5 throws");
}

// ========== Unit Tests: init_weights ==========

void test_init_weights() {
    std::cout << "[init_weights] dimensions and Xavier range..." << std::endl;
    auto w = init_weights(4, 16, 3, 42);

    ASSERT_TRUE(w.W1.size() == 16, "W1 rows=16");
    ASSERT_TRUE(w.W1[0].size() == 4, "W1 cols=4");
    ASSERT_TRUE(w.W2.size() == 3, "W2 rows=3");
    ASSERT_TRUE(w.W2[0].size() == 16, "W2 cols=16");
    ASSERT_TRUE(w.b1.size() == 16, "b1 size=16");
    ASSERT_TRUE(w.b2.size() == 3, "b2 size=3");

    // b1 and b2 should be zero
    for (double v : w.b1) {
        if (v != 0.0) { ASSERT_TRUE(false, "b1 not zero"); return; }
    }
    tests_passed++;
    for (double v : w.b2) {
        if (v != 0.0) { ASSERT_TRUE(false, "b2 not zero"); return; }
    }
    tests_passed++;

    // Xavier range for W1: sqrt(6/(4+16)) = sqrt(0.3)
    double limit1 = std::sqrt(6.0 / (4 + 16));
    bool in_range1 = true;
    for (const auto& row : w.W1) {
        for (double v : row) {
            if (v < -limit1 || v > limit1) { in_range1 = false; break; }
        }
    }
    ASSERT_TRUE(in_range1, "W1 in Xavier range");

    // Xavier range for W2: sqrt(6/(16+3)) = sqrt(6/19)
    double limit2 = std::sqrt(6.0 / (16 + 3));
    bool in_range2 = true;
    for (const auto& row : w.W2) {
        for (double v : row) {
            if (v < -limit2 || v > limit2) { in_range2 = false; break; }
        }
    }
    ASSERT_TRUE(in_range2, "W2 in Xavier range");

    // Deterministic
    auto w2 = init_weights(4, 16, 3, 42);
    ASSERT_NEAR(w.W1[0][0], w2.W1[0][0], 1e-15, "same seed -> same W1");
    ASSERT_NEAR(w.W2[0][0], w2.W2[0][0], 1e-15, "same seed -> same W2");

    // Throws on invalid
    ASSERT_THROWS(init_weights(0, 16, 3), "n_in=0 throws in init_weights");
}

// ========== Unit Tests: forward ==========

void test_forward() {
    std::cout << "[forward] shapes and softmax sum=1..." << std::endl;
    auto w = init_weights(4, 16, 3, 42);
    std::vector<double> x = {0.5, -0.3, 1.2, 0.8};

    auto cache = forward(x, w, relu);
    ASSERT_TRUE(cache.z1.size() == 16, "z1 size=16");
    ASSERT_TRUE(cache.h.size() == 16, "h size=16");
    ASSERT_TRUE(cache.z2.size() == 3, "z2 size=3");
    ASSERT_TRUE(cache.y.size() == 3, "y size=3");

    // y sums to 1 (softmax output)
    double y_sum = 0.0;
    for (double v : cache.y) y_sum += v;
    ASSERT_NEAR(y_sum, 1.0, 1e-10, "forward y sums to 1");

    // All y in [0,1]
    bool valid = true;
    for (double v : cache.y) {
        if (v < 0.0 || v > 1.0) { valid = false; break; }
    }
    ASSERT_TRUE(valid, "forward y values in [0,1]");

    // With ReLU, all h >= 0
    bool all_nonneg = true;
    for (double v : cache.h) {
        if (v < 0.0) { all_nonneg = false; break; }
    }
    ASSERT_TRUE(all_nonneg, "ReLU activations >= 0");

    // Test with sigmoid
    auto cache_sig = forward(x, w, sigmoid);
    ASSERT_TRUE(cache_sig.y.size() == 3, "sigmoid forward y size=3");
    double sig_sum = 0.0;
    for (double v : cache_sig.y) sig_sum += v;
    ASSERT_NEAR(sig_sum, 1.0, 1e-10, "sigmoid forward y sums to 1");
}

// ========== Unit Tests: cross_entropy ==========

void test_cross_entropy() {
    std::cout << "[cross_entropy] key values..." << std::endl;

    // Perfect prediction
    ASSERT_NEAR(cross_entropy({1.0, 0.0, 0.0}, 0), 0.0, 1e-10, "CE(1.0, 0) = 0");

    // Uniform prediction
    double expected_uniform = -std::log(1.0 / 3.0);
    ASSERT_NEAR(cross_entropy({1.0/3.0, 1.0/3.0, 1.0/3.0}, 0),
                expected_uniform, 1e-10, "CE(1/3, 0) = log(3)");

    // Very wrong prediction
    double expected_bad = -std::log(0.01);
    ASSERT_NEAR(cross_entropy({0.01, 0.98, 0.01}, 0),
                expected_bad, 1e-10, "CE(0.01, 0) = -log(0.01)");

    // Zero clipping
    double ce_zero = cross_entropy({0.0, 1.0, 0.0}, 0);
    ASSERT_TRUE(std::isfinite(ce_zero), "CE handles zero with clipping");

    // Out of range label
    ASSERT_THROWS(cross_entropy({0.5, 0.5}, -1), "CE label=-1 throws");
    ASSERT_THROWS(cross_entropy({0.5, 0.5}, 2), "CE label=2 throws");
}

// ========== Unit Tests: backward ==========

void test_backward() {
    std::cout << "[backward] shapes and dz2 sum..." << std::endl;
    auto w = init_weights(4, 16, 3, 42);
    std::vector<double> x = {0.5, -0.3, 1.2, 0.8};
    auto cache = forward(x, w, relu);
    auto grads = backward(x, 1, cache, w, relu_deriv);

    ASSERT_TRUE(grads.dW1.size() == 16, "dW1 rows=16");
    ASSERT_TRUE(grads.dW1[0].size() == 4, "dW1 cols=4");
    ASSERT_TRUE(grads.dW2.size() == 3, "dW2 rows=3");
    ASSERT_TRUE(grads.dW2[0].size() == 16, "dW2 cols=16");
    ASSERT_TRUE(grads.db1.size() == 16, "db1 size=16");
    ASSERT_TRUE(grads.db2.size() == 3, "db2 size=3");

    // dz2 = y - one_hot(label), sum of dz2 should be ~0
    // because sum(y) = 1 and sum(one_hot) = 1
    double dz2_sum = 0.0;
    for (double v : grads.db2) dz2_sum += v;
    ASSERT_NEAR(dz2_sum, 0.0, 1e-10, "dz2 sum ~ 0");

    // Test with different label
    auto grads2 = backward(x, 0, cache, w, relu_deriv);
    double dz2_sum2 = 0.0;
    for (double v : grads2.db2) dz2_sum2 += v;
    ASSERT_NEAR(dz2_sum2, 0.0, 1e-10, "dz2 sum ~ 0 (label=0)");

    // Gradients should differ for different labels
    bool differ = false;
    for (size_t k = 0; k < grads.db2.size(); ++k) {
        if (std::abs(grads.db2[k] - grads2.db2[k]) > 1e-10) { differ = true; break; }
    }
    ASSERT_TRUE(differ, "different labels -> different gradients");
}

// ========== Unit Tests: adam_update ==========

void test_adam_update() {
    std::cout << "[adam_update] weights change and t increments..." << std::endl;
    auto w = init_weights(4, 16, 3, 42);
    std::vector<double> x = {0.5, -0.3, 1.2, 0.8};
    auto cache = forward(x, w, relu);
    auto grads = backward(x, 1, cache, w, relu_deriv);
    auto state = init_adam(w);

    ASSERT_TRUE(state.t == 0, "initial t=0");

    double w1_before = w.W1[0][0];
    adam_update(w, grads, state);
    ASSERT_TRUE(state.t == 1, "t=1 after update");
    ASSERT_TRUE(std::abs(w.W1[0][0] - w1_before) > 1e-15, "W1 changed after update");

    double w1_after1 = w.W1[0][0];
    cache = forward(x, w, relu);
    grads = backward(x, 1, cache, w, relu_deriv);
    adam_update(w, grads, state);
    ASSERT_TRUE(state.t == 2, "t=2 after second update");
    ASSERT_TRUE(std::abs(w.W1[0][0] - w1_after1) > 1e-15, "W1 changed again");

    // Verify moment sizes
    ASSERT_TRUE(state.mW1.size() == 16, "mW1 rows=16");
    ASSERT_TRUE(state.vW1[0].size() == 4, "vW1 cols=4");
    ASSERT_TRUE(state.mW2.size() == 3, "mW2 rows=3");
    ASSERT_TRUE(state.mb1.size() == 16, "mb1 size=16");
    ASSERT_TRUE(state.mb2.size() == 3, "mb2 size=3");
}

// ========== Unit Tests: predict ==========

void test_predict() {
    std::cout << "[predict] valid class..." << std::endl;
    auto w = init_weights(4, 16, 3, 42);
    std::vector<double> x = {0.5, -0.3, 1.2, 0.8};

    int cls = predict(x, w, relu);
    ASSERT_TRUE(cls >= 0 && cls <= 2, "predict returns valid class (relu)");

    int cls2 = predict(x, w, sigmoid);
    ASSERT_TRUE(cls2 >= 0 && cls2 <= 2, "predict returns valid class (sigmoid)");

    int cls3 = predict(x, w, gelu);
    ASSERT_TRUE(cls3 >= 0 && cls3 <= 2, "predict returns valid class (gelu)");

    int cls4 = predict(x, w, bipolar_sigmoid);
    ASSERT_TRUE(cls4 >= 0 && cls4 <= 2, "predict returns valid class (bipolar)");
}

// ========== Integration: training convergence ==========

void test_training_convergence() {
    std::cout << "[integration] training converges (short)..." << std::endl;
    auto data = load_iris();
    std::vector<std::vector<double>> X(data.size());
    std::vector<int> y(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        X[i] = {data[i].features[0], data[i].features[1],
                 data[i].features[2], data[i].features[3]};
        y[i] = data[i].label;
    }

    auto split = split_data(X, y, 0.8, 42);
    auto norm = normalize(split.X_train, split.X_test);

    // Train with small epoch count for fast test
    Weights w = train(norm.X_train, split.y_train, 8, relu, relu_deriv, 50, 0.01, 42);
    double acc = accuracy(norm.X_train, split.y_train, w, relu);
    ASSERT_TRUE(acc > 0.5, "train accuracy > 50% after 50 epochs");

    double loss = compute_loss(norm.X_train, split.y_train, w, relu);
    ASSERT_TRUE(std::isfinite(loss), "finite loss after training");
    ASSERT_TRUE(loss > 0.0, "positive loss");
}

// ========== Main ==========

int main() {
    std::cout << "=== Running Tests ===" << std::endl << std::endl;

    // Unit: load_iris
    test_load_iris();

    // Unit: split_data
    test_split_data();

    // Unit: normalize
    test_normalize();

    // Unit: activations
    test_relu();
    test_sigmoid();
    test_bipolar_sigmoid();
    test_gelu();

    // Unit: softmax
    test_softmax();

    // Unit: validate_params
    test_validate_params();

    // Unit: init_weights
    test_init_weights();

    // Unit: forward
    test_forward();

    // Unit: cross_entropy
    test_cross_entropy();

    // Unit: backward
    test_backward();

    // Unit: adam_update
    test_adam_update();

    // Unit: predict
    test_predict();

    // Integration: training
    test_training_convergence();

    std::cout << std::endl << "=== Results ===" << std::endl;
    std::cout << "Passed: " << tests_passed << std::endl;
    std::cout << "Failed: " << tests_failed << std::endl;

    return tests_failed > 0 ? 1 : 0;
}
