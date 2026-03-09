#pragma once

#include <vector>
#include <random>
#include <cmath>
#include <stdexcept>
#include <numeric>
#include <algorithm>

constexpr double FREQ_MIN = 20.0;
constexpr double FREQ_MAX = 20000.0;

struct Params {
    int n_seconds;
    int bitrate;
    int sample_rate_khz;
};

inline bool is_valid_sample_rate(int rate) {
    const int valid_rates[] = {48, 96, 128, 176, 256};
    for (int r : valid_rates) {
        if (r == rate) return true;
    }
    return false;
}

inline Params validate_params(int n_seconds, int bitrate, int sample_rate_khz) {
    if (n_seconds <= 0) {
        throw std::invalid_argument("n должно быть > 0");
    }
    if (bitrate <= 0 || bitrate > 24) {
        throw std::invalid_argument("Битрейт должен быть от 1 до 24");
    }
    if (!is_valid_sample_rate(sample_rate_khz)) {
        throw std::invalid_argument("Недопустимая частота записи");
    }
    return {n_seconds, bitrate, sample_rate_khz};
}

inline std::vector<double> generate_signal(long long count, unsigned seed = 42) {
    std::vector<double> signal(count);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(FREQ_MIN, FREQ_MAX);

    for (long long i = 0; i < count; ++i) {
        signal[i] = dist(rng);
    }
    return signal;
}

inline double quantize(double value, double step) {
    double relative = value - FREQ_MIN;
    long long level = static_cast<long long>(std::round(relative / step));
    return FREQ_MIN + level * step;
}

inline double compute_step(int bitrate) {
    long long levels = (1LL << bitrate) - 1;
    return (FREQ_MAX - FREQ_MIN) / levels;
}

inline std::vector<double> discretize(const std::vector<double>& signal,
                                      int n_seconds,
                                      int bitrate,
                                      int sample_rate_khz) {
    long long total_input = static_cast<long long>(signal.size());
    long long sample_rate = static_cast<long long>(sample_rate_khz) * 1000;
    long long total_samples = sample_rate * n_seconds;

    double step = compute_step(bitrate);

    std::vector<double> output(total_samples);
    double samples_per_bin = static_cast<double>(total_input) / total_samples;

    for (long long i = 0; i < total_samples; ++i) {
        long long start = static_cast<long long>(i * samples_per_bin);
        long long end = static_cast<long long>((i + 1) * samples_per_bin);
        if (end > total_input) end = total_input;
        if (start >= end) start = end - 1;

        double sum = 0.0;
        for (long long j = start; j < end; ++j) {
            sum += signal[j];
        }
        double avg = sum / (end - start);

        output[i] = quantize(avg, step);
    }

    return output;
}
