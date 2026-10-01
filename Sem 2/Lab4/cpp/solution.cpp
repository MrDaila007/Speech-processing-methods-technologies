#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include "mlp.h"

struct ActivationEntry {
    std::string name;
    ActivationFn fn;
    ActivationFn deriv;
};

int main() {
    try {
        std::vector<ActivationEntry> activations = {
            {"ReLU",             relu,             relu_deriv},
            {"Sigmoid",          sigmoid,          sigmoid_deriv},
            {"Bipolar Sigmoid",  bipolar_sigmoid,  bipolar_sigmoid_deriv},
            {"GELU",             gelu,             gelu_deriv},
        };

        unsigned seed = 42;
        int n_hidden = HIDDEN_SIZE;
        int epochs = EPOCHS;
        double lr = LEARNING_RATE;

        // Load data
        auto samples = load_iris();
        std::vector<std::vector<double>> X(samples.size());
        std::vector<int> y(samples.size());
        for (size_t i = 0; i < samples.size(); ++i) {
            X[i] = {samples[i].features[0], samples[i].features[1],
                     samples[i].features[2], samples[i].features[3]};
            y[i] = samples[i].label;
        }

        // Split
        SplitResult split = split_data(X, y, TRAIN_RATIO, seed);

        // Normalize
        NormalizeResult norm = normalize(split.X_train, split.X_test);

        // Print header
        std::cout << "Датасет: Iris (150 образцов, 4 признака, 3 класса)" << std::endl;
        std::cout << "Скрытый слой: " << n_hidden << " нейронов" << std::endl;
        std::cout << "Оптимизатор: ADAM (lr=" << lr
                  << ", beta1=" << BETA1
                  << ", beta2=" << BETA2 << ")" << std::endl;
        std::cout << "Эпохи: " << epochs << std::endl;
        std::cout << std::endl;

        std::cout << "Сравнение функций активации скрытого слоя:" << std::endl;
        std::string sep = "+-------------------+------------------+-----------------+----------------+";
        std::cout << sep << std::endl;
        std::cout << "| Функция активации | Точность (обуч.) | Точность (тест) | Потери (обуч.) |" << std::endl;
        std::cout << sep << std::endl;

        for (const auto& act : activations) {
            Weights w = train(norm.X_train, split.y_train, n_hidden,
                              act.fn, act.deriv, epochs, lr, seed);

            double train_acc = accuracy(norm.X_train, split.y_train, w, act.fn);
            double test_acc  = accuracy(norm.X_test, split.y_test, w, act.fn);
            double train_loss = compute_loss(norm.X_train, split.y_train, w, act.fn);

            // Format name (left-aligned, 17 chars)
            std::string name = act.name;
            while (name.size() < 17) name += ' ';

            // Format accuracy (right-aligned)
            std::ostringstream oss_ta, oss_te, oss_lo;
            oss_ta << std::fixed << std::setprecision(2) << (train_acc * 100.0) << "%";
            oss_te << std::fixed << std::setprecision(2) << (test_acc * 100.0) << "%";
            oss_lo << std::fixed << std::setprecision(4) << train_loss;

            std::cout << "| " << name << " |"
                      << std::setw(16) << oss_ta.str() << " |"
                      << std::setw(15) << oss_te.str() << " |"
                      << std::setw(14) << oss_lo.str() << " |"
                      << std::endl;
        }
        std::cout << sep << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Ошибка: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
