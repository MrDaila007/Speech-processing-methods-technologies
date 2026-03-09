#include <iostream>
#include <iomanip>
#include "discretization.h"

Params read_params() {
    int n, br, sr;

    std::cout << "Количество секунд аудио (n): ";
    std::cin >> n;

    std::cout << "Битрейт (бит): ";
    std::cin >> br;

    std::cout << "Частота записи (48, 96, 128, 176, 256) KHz: ";
    std::cin >> sr;

    return validate_params(n, br, sr);
}

void print_results(const std::vector<double>& output,
                   int bitrate,
                   int sample_rate_khz,
                   int n_seconds) {
    long long levels = (1LL << bitrate) - 1;
    double step = compute_step(bitrate);
    long long total_samples = static_cast<long long>(sample_rate_khz) * 1000 * n_seconds;

    std::cout << "\n=== Результаты дискретизации ===" << std::endl;
    std::cout << "Частота записи:   " << sample_rate_khz << " KHz" << std::endl;
    std::cout << "Битрейт:          " << bitrate << " бит" << std::endl;
    std::cout << "Уровней квант.:   " << levels + 1 << std::endl;
    std::cout << "Шаг квантования:  " << std::fixed << std::setprecision(4) << step << " Hz" << std::endl;
    std::cout << "Всего сэмплов:    " << total_samples << std::endl;

    long long show = std::min(static_cast<long long>(output.size()), 20LL);
    std::cout << "\nПервые " << show << " сэмплов:" << std::endl;
    std::cout << std::setw(10) << "Сэмпл" << std::setw(16) << "Частота (Hz)" << std::endl;
    std::cout << std::string(26, '-') << std::endl;
    for (long long i = 0; i < show; ++i) {
        std::cout << std::setw(10) << i
                  << std::setw(16) << std::fixed << std::setprecision(4) << output[i]
                  << std::endl;
    }

    double out_min = *std::min_element(output.begin(), output.end());
    double out_max = *std::max_element(output.begin(), output.end());
    double out_avg = std::accumulate(output.begin(), output.end(), 0.0) / output.size();

    std::cout << "\nСтатистика:" << std::endl;
    std::cout << "  Min: " << std::setprecision(4) << out_min << " Hz" << std::endl;
    std::cout << "  Max: " << std::setprecision(4) << out_max << " Hz" << std::endl;
    std::cout << "  Avg: " << std::setprecision(4) << out_avg << " Hz" << std::endl;
}

int main() {
    try {
        Params p = read_params();

        long long total_values = 1000000LL * p.n_seconds;
        std::cout << "\nГенерация " << total_values << " значений сигнала..." << std::endl;
        std::vector<double> signal = generate_signal(total_values);

        std::cout << "Дискретизация..." << std::endl;
        std::vector<double> output = discretize(signal, p.n_seconds, p.bitrate, p.sample_rate_khz);

        print_results(output, p.bitrate, p.sample_rate_khz, p.n_seconds);

    } catch (const std::exception& e) {
        std::cerr << "Ошибка: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
