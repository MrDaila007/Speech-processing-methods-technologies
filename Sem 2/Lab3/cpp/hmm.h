#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <stdexcept>
#include <numeric>
#include <limits>

constexpr int N_OBSERVATIONS = 27;

struct HMM {
    int n_states;
    std::vector<double> pi;
    std::vector<std::vector<double>> A;
    std::vector<std::vector<double>> B;
};

struct ForwardResult {
    double log_probability;
};

inline void validate_params(int n_states) {
    if (n_states <= 0) {
        throw std::invalid_argument("Число состояний должно быть > 0");
    }
}

inline int char_to_index(char c) {
    if (c >= 'a' && c <= 'z') {
        return c - 'a';
    }
    if (c == ' ') {
        return 26;
    }
    throw std::invalid_argument("Недопустимый символ: допустимы только a-z и пробел");
}

inline std::vector<int> text_to_indices(const std::string& text) {
    if (text.empty()) {
        throw std::invalid_argument("Текст не должен быть пустым");
    }
    std::vector<int> indices;
    indices.reserve(text.size());
    for (char c : text) {
        indices.push_back(char_to_index(c));
    }
    return indices;
}

inline std::vector<double> normalize(const std::vector<double>& values) {
    double sum = 0.0;
    for (double v : values) {
        sum += v;
    }
    if (sum <= 0.0) {
        throw std::invalid_argument("Невозможно нормализовать: сумма <= 0");
    }
    std::vector<double> result(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        result[i] = values[i] / sum;
    }
    return result;
}

inline HMM init_hmm(int n_states, unsigned seed = 42) {
    validate_params(n_states);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    HMM hmm;
    hmm.n_states = n_states;

    std::vector<double> raw(n_states);
    for (int i = 0; i < n_states; ++i) {
        raw[i] = dist(rng);
    }
    hmm.pi = normalize(raw);

    hmm.A.resize(n_states);
    for (int i = 0; i < n_states; ++i) {
        std::vector<double> row(n_states);
        for (int j = 0; j < n_states; ++j) {
            row[j] = dist(rng);
        }
        hmm.A[i] = normalize(row);
    }

    hmm.B.resize(n_states);
    for (int i = 0; i < n_states; ++i) {
        std::vector<double> row(N_OBSERVATIONS);
        for (int j = 0; j < N_OBSERVATIONS; ++j) {
            row[j] = dist(rng);
        }
        hmm.B[i] = normalize(row);
    }

    return hmm;
}

inline ForwardResult forward(const std::vector<int>& observations, const HMM& hmm) {
    if (observations.empty()) {
        throw std::invalid_argument("Последовательность наблюдений не должна быть пустой");
    }
    for (int obs : observations) {
        if (obs < 0 || obs >= N_OBSERVATIONS) {
            throw std::invalid_argument("Индекс наблюдения вне допустимого диапазона");
        }
    }

    int T = static_cast<int>(observations.size());
    int N = hmm.n_states;

    // Scaled forward algorithm (Rabiner)
    std::vector<std::vector<double>> alpha(T, std::vector<double>(N));
    std::vector<double> scaling(T);

    // Initialization
    double scale = 0.0;
    for (int i = 0; i < N; ++i) {
        alpha[0][i] = hmm.pi[i] * hmm.B[i][observations[0]];
        scale += alpha[0][i];
    }
    if (scale <= 0.0) {
        return {-std::numeric_limits<double>::infinity()};
    }
    scaling[0] = 1.0 / scale;
    for (int i = 0; i < N; ++i) {
        alpha[0][i] *= scaling[0];
    }

    // Induction
    for (int t = 1; t < T; ++t) {
        scale = 0.0;
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int i = 0; i < N; ++i) {
                sum += alpha[t - 1][i] * hmm.A[i][j];
            }
            alpha[t][j] = sum * hmm.B[j][observations[t]];
            scale += alpha[t][j];
        }
        if (scale <= 0.0) {
            return {-std::numeric_limits<double>::infinity()};
        }
        scaling[t] = 1.0 / scale;
        for (int j = 0; j < N; ++j) {
            alpha[t][j] *= scaling[t];
        }
    }

    // Termination: log P(O|λ) = -Σ log(c_t)
    double log_prob = 0.0;
    for (int t = 0; t < T; ++t) {
        log_prob -= std::log(scaling[t]);
    }

    return {log_prob};
}
