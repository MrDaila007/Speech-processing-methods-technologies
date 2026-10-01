#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include "hmm.h"

void print_results(const HMM& hmm,
                   const std::vector<std::string>& phrases,
                   const std::vector<ForwardResult>& results) {
    std::cout << "\n=== Результаты СММ ===" << std::endl;
    std::cout << "Скрытых состояний:  " << hmm.n_states << std::endl;
    std::cout << "Символов наблюдения: " << N_OBSERVATIONS
              << " (a-z + пробел)" << std::endl;

    std::cout << "\nВероятности фраз:" << std::endl;
    std::cout << std::setw(30) << "Фраза"
              << std::setw(10) << "Длина"
              << std::setw(20) << "Лог-вероятность" << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    for (size_t i = 0; i < phrases.size(); ++i) {
        std::string display = "\"" + phrases[i] + "\"";
        std::cout << std::setw(30) << display
                  << std::setw(10) << phrases[i].size()
                  << std::setw(20) << std::fixed << std::setprecision(4)
                  << results[i].log_probability << std::endl;
    }
}

int main() {
    try {
        int n_states;
        std::cout << "Число скрытых состояний: ";
        std::cin >> n_states;
        std::cin.ignore();

        std::cout << "Введите фразы (пустая строка для завершения):" << std::endl;
        std::vector<std::string> phrases;
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) break;
            phrases.push_back(line);
        }

        if (phrases.empty()) {
            std::cerr << "Ошибка: не введено ни одной фразы" << std::endl;
            return 1;
        }

        std::cout << "\nИнициализация СММ..." << std::endl;
        HMM hmm = init_hmm(n_states);

        std::cout << "Вычисление вероятностей..." << std::endl;
        std::vector<ForwardResult> results;
        for (const auto& phrase : phrases) {
            auto obs = text_to_indices(phrase);
            results.push_back(forward(obs, hmm));
        }

        print_results(hmm, phrases, results);

    } catch (const std::exception& e) {
        std::cerr << "Ошибка: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
