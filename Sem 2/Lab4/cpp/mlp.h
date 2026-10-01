#pragma once

#include <vector>
#include <cmath>
#include <random>
#include <stdexcept>
#include <algorithm>
#include <utility>

// ── Constants ────────────────────────────────────────────────────────

constexpr int HIDDEN_SIZE = 16;
constexpr double LEARNING_RATE = 0.001;
constexpr int EPOCHS = 500;
constexpr double BETA1 = 0.9;
constexpr double BETA2 = 0.999;
constexpr double ADAM_EPSILON = 1e-8;
constexpr double TRAIN_RATIO = 0.8;

// ── Structs ──────────────────────────────────────────────────────────

struct IrisSample {
    double features[4];
    int label;
};

struct Weights {
    std::vector<std::vector<double>> W1, W2;
    std::vector<double> b1, b2;
};

struct ForwardCache {
    std::vector<double> z1, h, z2, y;
};

struct Gradients {
    std::vector<std::vector<double>> dW1, dW2;
    std::vector<double> db1, db2;
};

struct AdamState {
    std::vector<std::vector<double>> mW1, vW1, mW2, vW2;
    std::vector<double> mb1, vb1, mb2, vb2;
    int t;
};

struct TrainResult {
    double train_acc, test_acc, train_loss;
};

// ── Activation function pointer type ─────────────────────────────────

using ActivationFn = double(*)(double);

// ── Iris Dataset (Fisher, 1936) ──────────────────────────────────────

inline std::vector<IrisSample> load_iris() {
    return {
        // Setosa (class 0)
        {{5.1,3.5,1.4,0.2}, 0}, {{4.9,3.0,1.4,0.2}, 0}, {{4.7,3.2,1.3,0.2}, 0},
        {{4.6,3.1,1.5,0.2}, 0}, {{5.0,3.6,1.4,0.2}, 0}, {{5.4,3.9,1.7,0.4}, 0},
        {{4.6,3.4,1.4,0.3}, 0}, {{5.0,3.4,1.5,0.2}, 0}, {{4.4,2.9,1.4,0.2}, 0},
        {{4.9,3.1,1.5,0.1}, 0}, {{5.4,3.7,1.5,0.2}, 0}, {{4.8,3.4,1.6,0.2}, 0},
        {{4.8,3.0,1.4,0.1}, 0}, {{4.3,3.0,1.1,0.1}, 0}, {{5.8,4.0,1.2,0.2}, 0},
        {{5.7,4.4,1.5,0.4}, 0}, {{5.4,3.9,1.3,0.4}, 0}, {{5.1,3.5,1.4,0.3}, 0},
        {{5.7,3.8,1.7,0.3}, 0}, {{5.1,3.8,1.5,0.3}, 0}, {{5.4,3.4,1.7,0.2}, 0},
        {{5.1,3.7,1.5,0.4}, 0}, {{4.6,3.6,1.0,0.2}, 0}, {{5.1,3.3,1.7,0.5}, 0},
        {{4.8,3.4,1.9,0.2}, 0}, {{5.0,3.0,1.6,0.2}, 0}, {{5.0,3.4,1.6,0.4}, 0},
        {{5.2,3.5,1.5,0.2}, 0}, {{5.2,3.4,1.4,0.2}, 0}, {{4.7,3.2,1.6,0.2}, 0},
        {{4.8,3.1,1.6,0.2}, 0}, {{5.4,3.4,1.5,0.4}, 0}, {{5.2,4.1,1.5,0.1}, 0},
        {{5.5,4.2,1.4,0.2}, 0}, {{4.9,3.1,1.5,0.2}, 0}, {{5.0,3.2,1.2,0.2}, 0},
        {{5.5,3.5,1.3,0.2}, 0}, {{4.9,3.6,1.4,0.1}, 0}, {{4.4,3.0,1.3,0.2}, 0},
        {{5.1,3.4,1.5,0.2}, 0}, {{5.0,3.5,1.3,0.3}, 0}, {{4.5,2.3,1.3,0.3}, 0},
        {{4.4,3.2,1.3,0.2}, 0}, {{5.0,3.5,1.6,0.6}, 0}, {{5.1,3.8,1.9,0.4}, 0},
        {{4.8,3.0,1.4,0.3}, 0}, {{5.1,3.8,1.6,0.2}, 0}, {{4.6,3.2,1.4,0.2}, 0},
        {{5.3,3.7,1.5,0.2}, 0}, {{5.0,3.3,1.4,0.2}, 0},
        // Versicolor (class 1)
        {{7.0,3.2,4.7,1.4}, 1}, {{6.4,3.2,4.5,1.5}, 1}, {{6.9,3.1,4.9,1.5}, 1},
        {{5.5,2.3,4.0,1.3}, 1}, {{6.5,2.8,4.6,1.5}, 1}, {{5.7,2.8,4.5,1.3}, 1},
        {{6.3,3.3,4.7,1.6}, 1}, {{4.9,2.4,3.3,1.0}, 1}, {{6.6,2.9,4.6,1.3}, 1},
        {{5.2,2.7,3.9,1.4}, 1}, {{5.0,2.0,3.5,1.0}, 1}, {{5.9,3.0,4.2,1.5}, 1},
        {{6.0,2.2,4.0,1.0}, 1}, {{6.1,2.9,4.7,1.4}, 1}, {{5.6,2.9,3.6,1.3}, 1},
        {{6.7,3.1,4.4,1.4}, 1}, {{5.6,3.0,4.5,1.5}, 1}, {{5.8,2.7,4.1,1.0}, 1},
        {{6.2,2.2,4.5,1.5}, 1}, {{5.6,2.5,3.9,1.1}, 1}, {{5.9,3.2,4.8,1.8}, 1},
        {{6.1,2.8,4.0,1.3}, 1}, {{6.3,2.5,4.9,1.5}, 1}, {{6.1,2.8,4.7,1.2}, 1},
        {{6.4,2.9,4.3,1.3}, 1}, {{6.6,3.0,4.4,1.4}, 1}, {{6.8,2.8,4.8,1.4}, 1},
        {{6.7,3.0,5.0,1.7}, 1}, {{6.0,2.9,4.5,1.5}, 1}, {{5.7,2.6,3.5,1.0}, 1},
        {{5.5,2.4,3.8,1.1}, 1}, {{5.5,2.4,3.7,1.0}, 1}, {{5.8,2.7,3.9,1.2}, 1},
        {{6.0,2.7,5.1,1.6}, 1}, {{5.4,3.0,4.5,1.5}, 1}, {{6.0,3.4,4.5,1.6}, 1},
        {{6.7,3.1,4.7,1.5}, 1}, {{6.3,2.3,4.4,1.3}, 1}, {{5.6,3.0,4.1,1.3}, 1},
        {{5.5,2.5,4.0,1.3}, 1}, {{5.5,2.6,4.4,1.2}, 1}, {{6.1,3.0,4.6,1.4}, 1},
        {{5.8,2.6,4.0,1.2}, 1}, {{5.0,2.3,3.3,1.0}, 1}, {{5.6,2.7,4.2,1.3}, 1},
        {{5.7,3.0,4.2,1.2}, 1}, {{5.7,2.9,4.2,1.3}, 1}, {{6.2,2.9,4.3,1.3}, 1},
        {{5.1,2.5,3.0,1.1}, 1}, {{5.7,2.8,4.1,1.3}, 1},
        // Virginica (class 2)
        {{6.3,3.3,6.0,2.5}, 2}, {{5.8,2.7,5.1,1.9}, 2}, {{7.1,3.0,5.9,2.1}, 2},
        {{6.3,2.9,5.6,1.8}, 2}, {{6.5,3.0,5.8,2.2}, 2}, {{7.6,3.0,6.6,2.1}, 2},
        {{4.9,2.5,4.5,1.7}, 2}, {{7.3,2.9,6.3,1.8}, 2}, {{6.7,2.5,5.8,1.8}, 2},
        {{7.2,3.6,6.1,2.5}, 2}, {{6.5,3.2,5.1,2.0}, 2}, {{6.4,2.7,5.3,1.9}, 2},
        {{6.8,3.0,5.5,2.1}, 2}, {{5.7,2.5,5.0,2.0}, 2}, {{5.8,2.8,5.1,2.4}, 2},
        {{6.4,3.2,5.3,2.3}, 2}, {{6.5,3.0,5.5,1.8}, 2}, {{7.7,3.8,6.7,2.2}, 2},
        {{7.7,2.6,6.9,2.3}, 2}, {{6.0,2.2,5.0,1.5}, 2}, {{6.9,3.2,5.7,2.3}, 2},
        {{5.6,2.8,4.9,2.0}, 2}, {{7.7,2.8,6.7,2.0}, 2}, {{6.3,2.7,4.9,1.8}, 2},
        {{6.7,3.3,5.7,2.1}, 2}, {{7.2,3.2,6.0,1.8}, 2}, {{6.2,2.8,4.8,1.8}, 2},
        {{6.1,3.0,4.9,1.8}, 2}, {{6.4,2.8,5.6,2.1}, 2}, {{7.2,3.0,5.8,1.6}, 2},
        {{7.4,2.8,6.1,1.9}, 2}, {{7.9,3.8,6.4,2.0}, 2}, {{6.4,2.8,5.6,2.2}, 2},
        {{6.3,2.8,5.1,1.5}, 2}, {{6.1,2.6,5.6,1.4}, 2}, {{7.7,3.0,6.1,2.3}, 2},
        {{6.3,3.4,5.6,2.4}, 2}, {{6.4,3.1,5.5,1.8}, 2}, {{6.0,3.0,4.8,1.8}, 2},
        {{6.9,3.1,5.4,2.1}, 2}, {{6.7,3.1,5.6,2.4}, 2}, {{6.9,3.1,5.1,2.3}, 2},
        {{5.8,2.7,5.1,1.9}, 2}, {{6.8,3.2,5.9,2.3}, 2}, {{6.7,3.3,5.7,2.5}, 2},
        {{6.7,3.0,5.2,2.3}, 2}, {{6.3,2.5,5.0,1.9}, 2}, {{6.5,3.0,5.2,2.0}, 2},
        {{6.2,3.4,5.4,2.3}, 2}, {{5.9,3.0,5.1,1.8}, 2},
    };
}

// ── Data Splitting (Fisher-Yates shuffle) ────────────────────────────

struct SplitResult {
    std::vector<std::vector<double>> X_train, X_test;
    std::vector<int> y_train, y_test;
};

inline SplitResult split_data(const std::vector<std::vector<double>>& X,
                              const std::vector<int>& y,
                              double ratio, unsigned seed = 42) {
    if (X.empty()) {
        throw std::invalid_argument("Набор данных не может быть пустым");
    }
    if (ratio <= 0.0 || ratio >= 1.0) {
        throw std::invalid_argument("Коэффициент разделения должен быть в (0, 1)");
    }

    int n = static_cast<int>(X.size());
    std::vector<int> indices(n);
    for (int i = 0; i < n; ++i) indices[i] = i;

    // Fisher-Yates shuffle
    std::mt19937 rng(seed);
    for (int i = n - 1; i > 0; --i) {
        std::uniform_int_distribution<int> dist(0, i);
        int j = dist(rng);
        std::swap(indices[i], indices[j]);
    }

    int n_train = static_cast<int>(n * ratio);
    SplitResult result;
    result.X_train.reserve(n_train);
    result.y_train.reserve(n_train);
    result.X_test.reserve(n - n_train);
    result.y_test.reserve(n - n_train);

    for (int i = 0; i < n_train; ++i) {
        result.X_train.push_back(X[indices[i]]);
        result.y_train.push_back(y[indices[i]]);
    }
    for (int i = n_train; i < n; ++i) {
        result.X_test.push_back(X[indices[i]]);
        result.y_test.push_back(y[indices[i]]);
    }
    return result;
}

// ── Normalization (Z-score) ──────────────────────────────────────────

struct NormalizeResult {
    std::vector<std::vector<double>> X_train, X_test;
};

inline NormalizeResult normalize(const std::vector<std::vector<double>>& X_train,
                                 const std::vector<std::vector<double>>& X_test) {
    if (X_train.empty()) {
        throw std::invalid_argument("Обучающий набор не может быть пустым");
    }
    int n = static_cast<int>(X_train.size());
    int nf = static_cast<int>(X_train[0].size());

    std::vector<double> means(nf, 0.0);
    for (int j = 0; j < nf; ++j) {
        for (int i = 0; i < n; ++i) {
            means[j] += X_train[i][j];
        }
        means[j] /= n;
    }

    std::vector<double> stds(nf, 0.0);
    for (int j = 0; j < nf; ++j) {
        for (int i = 0; i < n; ++i) {
            double d = X_train[i][j] - means[j];
            stds[j] += d * d;
        }
        stds[j] = std::sqrt(stds[j] / n);
        if (stds[j] < 1e-12) stds[j] = 1.0;
    }

    NormalizeResult result;
    result.X_train.resize(X_train.size());
    for (size_t i = 0; i < X_train.size(); ++i) {
        result.X_train[i].resize(nf);
        for (int j = 0; j < nf; ++j) {
            result.X_train[i][j] = (X_train[i][j] - means[j]) / stds[j];
        }
    }
    result.X_test.resize(X_test.size());
    for (size_t i = 0; i < X_test.size(); ++i) {
        result.X_test[i].resize(nf);
        for (int j = 0; j < nf; ++j) {
            result.X_test[i][j] = (X_test[i][j] - means[j]) / stds[j];
        }
    }
    return result;
}

// ── Activation Functions ─────────────────────────────────────────────

inline double safe_sigmoid(double z) {
    z = std::max(-500.0, std::min(500.0, z));
    if (z >= 0.0) {
        return 1.0 / (1.0 + std::exp(-z));
    }
    double ez = std::exp(z);
    return ez / (1.0 + ez);
}

inline double relu(double z) {
    return std::max(0.0, z);
}

inline double relu_deriv(double z) {
    return z > 0.0 ? 1.0 : 0.0;
}

inline double sigmoid(double z) {
    return safe_sigmoid(z);
}

inline double sigmoid_deriv(double z) {
    double s = safe_sigmoid(z);
    return s * (1.0 - s);
}

inline double bipolar_sigmoid(double z) {
    return 2.0 * safe_sigmoid(z) - 1.0;
}

inline double bipolar_sigmoid_deriv(double z) {
    double s = safe_sigmoid(z);
    return 2.0 * s * (1.0 - s);
}

inline double gelu(double z) {
    double c = std::sqrt(2.0 / M_PI);
    double u = c * (z + 0.044715 * z * z * z);
    return 0.5 * z * (1.0 + std::tanh(u));
}

inline double gelu_deriv(double z) {
    double c = std::sqrt(2.0 / M_PI);
    double u = c * (z + 0.044715 * z * z * z);
    double du = c * (1.0 + 0.134145 * z * z);
    double tanh_u = std::tanh(u);
    return 0.5 * (1.0 + tanh_u) + 0.5 * z * (1.0 - tanh_u * tanh_u) * du;
}

// ── Softmax (with max subtraction for numerical stability) ───────────

inline std::vector<double> softmax(const std::vector<double>& z) {
    if (z.empty()) {
        throw std::invalid_argument("Вектор softmax не может быть пустым");
    }
    double m = *std::max_element(z.begin(), z.end());
    std::vector<double> e(z.size());
    double s = 0.0;
    for (size_t i = 0; i < z.size(); ++i) {
        e[i] = std::exp(z[i] - m);
        s += e[i];
    }
    for (size_t i = 0; i < e.size(); ++i) {
        e[i] /= s;
    }
    return e;
}

// ── Parameter Validation ─────────────────────────────────────────────

inline void validate_params(int n_in, int n_hidden, int n_out) {
    if (n_in <= 0) {
        throw std::invalid_argument("Число входов должно быть положительным");
    }
    if (n_hidden <= 0) {
        throw std::invalid_argument("Число нейронов скрытого слоя должно быть положительным");
    }
    if (n_out <= 0) {
        throw std::invalid_argument("Число выходов должно быть положительным");
    }
}

// ── Weight Initialization (Xavier/Glorot uniform) ────────────────────

inline Weights init_weights(int n_in, int n_hidden, int n_out, unsigned seed = 42) {
    validate_params(n_in, n_hidden, n_out);
    std::mt19937 rng(seed);

    double l1 = std::sqrt(6.0 / (n_in + n_hidden));
    double l2 = std::sqrt(6.0 / (n_hidden + n_out));

    std::uniform_real_distribution<double> dist1(-l1, l1);
    std::uniform_real_distribution<double> dist2(-l2, l2);

    Weights w;
    w.W1.resize(n_hidden, std::vector<double>(n_in));
    for (int i = 0; i < n_hidden; ++i) {
        for (int j = 0; j < n_in; ++j) {
            w.W1[i][j] = dist1(rng);
        }
    }
    w.b1.assign(n_hidden, 0.0);

    w.W2.resize(n_out, std::vector<double>(n_hidden));
    for (int k = 0; k < n_out; ++k) {
        for (int i = 0; i < n_hidden; ++i) {
            w.W2[k][i] = dist2(rng);
        }
    }
    w.b2.assign(n_out, 0.0);

    return w;
}

// ── Forward Pass ─────────────────────────────────────────────────────

inline ForwardCache forward(const std::vector<double>& x,
                            const Weights& weights,
                            ActivationFn act_fn) {
    int ni = static_cast<int>(x.size());
    int nh = static_cast<int>(weights.b1.size());
    int no = static_cast<int>(weights.b2.size());

    ForwardCache cache;
    cache.z1.resize(nh);
    cache.h.resize(nh);
    cache.z2.resize(no);

    // Hidden layer
    for (int i = 0; i < nh; ++i) {
        double sum = weights.b1[i];
        for (int j = 0; j < ni; ++j) {
            sum += weights.W1[i][j] * x[j];
        }
        cache.z1[i] = sum;
        cache.h[i] = act_fn(sum);
    }

    // Output layer
    for (int k = 0; k < no; ++k) {
        double sum = weights.b2[k];
        for (int i = 0; i < nh; ++i) {
            sum += weights.W2[k][i] * cache.h[i];
        }
        cache.z2[k] = sum;
    }

    cache.y = softmax(cache.z2);
    return cache;
}

// ── Cross-Entropy Loss ───────────────────────────────────────────────

inline double cross_entropy(const std::vector<double>& y_pred, int label) {
    if (label < 0 || label >= static_cast<int>(y_pred.size())) {
        throw std::invalid_argument("Индекс класса вне допустимого диапазона");
    }
    return -std::log(std::max(y_pred[label], 1e-15));
}

// ── Backward Pass ────────────────────────────────────────────────────

inline Gradients backward(const std::vector<double>& x,
                          int label,
                          const ForwardCache& cache,
                          const Weights& weights,
                          ActivationFn act_deriv) {
    int ni = static_cast<int>(x.size());
    int nh = static_cast<int>(cache.z1.size());
    int no = static_cast<int>(cache.y.size());

    Gradients grads;

    // Output layer gradients: dz2 = y - one_hot(label)
    std::vector<double> dz2(no);
    for (int k = 0; k < no; ++k) {
        dz2[k] = cache.y[k] - (k == label ? 1.0 : 0.0);
    }

    grads.dW2.resize(no, std::vector<double>(nh));
    for (int k = 0; k < no; ++k) {
        for (int i = 0; i < nh; ++i) {
            grads.dW2[k][i] = dz2[k] * cache.h[i];
        }
    }
    grads.db2 = dz2;

    // Hidden layer gradients
    std::vector<double> dh(nh, 0.0);
    for (int i = 0; i < nh; ++i) {
        for (int k = 0; k < no; ++k) {
            dh[i] += weights.W2[k][i] * dz2[k];
        }
    }

    std::vector<double> dz1(nh);
    for (int i = 0; i < nh; ++i) {
        dz1[i] = dh[i] * act_deriv(cache.z1[i]);
    }

    grads.dW1.resize(nh, std::vector<double>(ni));
    for (int i = 0; i < nh; ++i) {
        for (int j = 0; j < ni; ++j) {
            grads.dW1[i][j] = dz1[i] * x[j];
        }
    }
    grads.db1 = dz1;

    return grads;
}

// ── ADAM Optimizer ───────────────────────────────────────────────────

inline std::vector<std::vector<double>> zeros_mat(int rows, int cols) {
    return std::vector<std::vector<double>>(rows, std::vector<double>(cols, 0.0));
}

inline AdamState init_adam(const Weights& weights) {
    int nh = static_cast<int>(weights.W1.size());
    int ni = static_cast<int>(weights.W1[0].size());
    int no = static_cast<int>(weights.W2.size());

    AdamState state;
    state.mW1 = zeros_mat(nh, ni);
    state.vW1 = zeros_mat(nh, ni);
    state.mb1.assign(nh, 0.0);
    state.vb1.assign(nh, 0.0);
    state.mW2 = zeros_mat(no, nh);
    state.vW2 = zeros_mat(no, nh);
    state.mb2.assign(no, 0.0);
    state.vb2.assign(no, 0.0);
    state.t = 0;
    return state;
}

inline void adam_update(Weights& weights, const Gradients& grads,
                        AdamState& state, double lr = LEARNING_RATE) {
    state.t += 1;
    double bc1 = 1.0 - std::pow(BETA1, state.t);
    double bc2 = 1.0 - std::pow(BETA2, state.t);

    // Update W1
    for (size_t i = 0; i < weights.W1.size(); ++i) {
        for (size_t j = 0; j < weights.W1[0].size(); ++j) {
            state.mW1[i][j] = BETA1 * state.mW1[i][j] + (1.0 - BETA1) * grads.dW1[i][j];
            state.vW1[i][j] = BETA2 * state.vW1[i][j] + (1.0 - BETA2) * grads.dW1[i][j] * grads.dW1[i][j];
            weights.W1[i][j] -= lr * (state.mW1[i][j] / bc1) / (std::sqrt(state.vW1[i][j] / bc2) + ADAM_EPSILON);
        }
    }

    // Update W2
    for (size_t i = 0; i < weights.W2.size(); ++i) {
        for (size_t j = 0; j < weights.W2[0].size(); ++j) {
            state.mW2[i][j] = BETA1 * state.mW2[i][j] + (1.0 - BETA1) * grads.dW2[i][j];
            state.vW2[i][j] = BETA2 * state.vW2[i][j] + (1.0 - BETA2) * grads.dW2[i][j] * grads.dW2[i][j];
            weights.W2[i][j] -= lr * (state.mW2[i][j] / bc1) / (std::sqrt(state.vW2[i][j] / bc2) + ADAM_EPSILON);
        }
    }

    // Update b1
    for (size_t i = 0; i < weights.b1.size(); ++i) {
        state.mb1[i] = BETA1 * state.mb1[i] + (1.0 - BETA1) * grads.db1[i];
        state.vb1[i] = BETA2 * state.vb1[i] + (1.0 - BETA2) * grads.db1[i] * grads.db1[i];
        weights.b1[i] -= lr * (state.mb1[i] / bc1) / (std::sqrt(state.vb1[i] / bc2) + ADAM_EPSILON);
    }

    // Update b2
    for (size_t i = 0; i < weights.b2.size(); ++i) {
        state.mb2[i] = BETA1 * state.mb2[i] + (1.0 - BETA1) * grads.db2[i];
        state.vb2[i] = BETA2 * state.vb2[i] + (1.0 - BETA2) * grads.db2[i] * grads.db2[i];
        weights.b2[i] -= lr * (state.mb2[i] / bc1) / (std::sqrt(state.vb2[i] / bc2) + ADAM_EPSILON);
    }
}

// ── Training ─────────────────────────────────────────────────────────

inline Weights train(const std::vector<std::vector<double>>& X_train,
                     const std::vector<int>& y_train,
                     int n_hidden,
                     ActivationFn act_fn,
                     ActivationFn act_deriv,
                     int epochs = EPOCHS,
                     double lr = LEARNING_RATE,
                     unsigned seed = 42) {
    if (X_train.empty()) {
        throw std::invalid_argument("Обучающий набор не может быть пустым");
    }
    int n_input = static_cast<int>(X_train[0].size());
    int n_output = *std::max_element(y_train.begin(), y_train.end()) + 1;

    Weights weights = init_weights(n_input, n_hidden, n_output, seed);
    AdamState state = init_adam(weights);

    std::mt19937 rng(seed + 1);
    int n = static_cast<int>(X_train.size());

    for (int ep = 0; ep < epochs; ++ep) {
        std::vector<int> idx(n);
        for (int i = 0; i < n; ++i) idx[i] = i;

        // Shuffle indices
        for (int i = n - 1; i > 0; --i) {
            std::uniform_int_distribution<int> dist(0, i);
            int j = dist(rng);
            std::swap(idx[i], idx[j]);
        }

        for (int ii = 0; ii < n; ++ii) {
            int i = idx[ii];
            ForwardCache cache = forward(X_train[i], weights, act_fn);
            Gradients grads = backward(X_train[i], y_train[i], cache, weights, act_deriv);
            adam_update(weights, grads, state, lr);
        }
    }
    return weights;
}

// ── Prediction & Evaluation ──────────────────────────────────────────

inline int predict(const std::vector<double>& x,
                   const Weights& weights,
                   ActivationFn act_fn) {
    ForwardCache cache = forward(x, weights, act_fn);
    return static_cast<int>(std::max_element(cache.y.begin(), cache.y.end()) - cache.y.begin());
}

inline double accuracy(const std::vector<std::vector<double>>& X,
                       const std::vector<int>& y,
                       const Weights& weights,
                       ActivationFn act_fn) {
    if (X.empty()) {
        throw std::invalid_argument("Набор данных не может быть пустым");
    }
    int correct = 0;
    for (size_t i = 0; i < X.size(); ++i) {
        if (predict(X[i], weights, act_fn) == y[i]) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / static_cast<double>(X.size());
}

inline double compute_loss(const std::vector<std::vector<double>>& X,
                           const std::vector<int>& y,
                           const Weights& weights,
                           ActivationFn act_fn) {
    if (X.empty()) {
        throw std::invalid_argument("Набор данных не может быть пустым");
    }
    double total = 0.0;
    for (size_t i = 0; i < X.size(); ++i) {
        ForwardCache cache = forward(X[i], weights, act_fn);
        total += cross_entropy(cache.y, y[i]);
    }
    return total / static_cast<double>(X.size());
}
