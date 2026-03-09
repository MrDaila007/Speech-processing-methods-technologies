#include <iostream>
#include <iomanip>
#include "kmeans.h"

void print_results(const KMeansResult& result, const std::vector<Point>& points) {
    std::cout << "\n=== Результаты кластеризации ===" << std::endl;
    std::cout << "Итераций:            " << result.iterations << std::endl;
    std::cout << "Суммарное расстояние: " << std::fixed << std::setprecision(2)
              << result.total_distance << std::endl;
    std::cout << "Доля ошибок:         " << std::setprecision(4)
              << result.error_rate * 100.0 << "%" << std::endl;

    std::cout << "\nКластеры:" << std::endl;
    std::cout << std::setw(10) << "Кластер"
              << std::setw(10) << "Класс"
              << std::setw(10) << "Точек"
              << std::setw(12) << "Ошибок"
              << std::setw(14) << "Центр X"
              << std::setw(14) << "Центр Y" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    for (const auto& ci : result.cluster_info) {
        std::cout << std::setw(10) << ci.cluster_id
                  << std::setw(10) << ci.assigned_class
                  << std::setw(10) << ci.total_points
                  << std::setw(12) << ci.mismatched_points
                  << std::setw(14) << std::setprecision(4) << result.centroids[ci.cluster_id].x
                  << std::setw(14) << result.centroids[ci.cluster_id].y
                  << std::endl;
    }

    int show = std::min(static_cast<int>(points.size()), 20);
    std::cout << "\nПервые " << show << " точек:" << std::endl;
    std::cout << std::setw(8) << "#"
              << std::setw(12) << "X"
              << std::setw(12) << "Y"
              << std::setw(10) << "Класс"
              << std::setw(12) << "Кластер" << std::endl;
    std::cout << std::string(54, '-') << std::endl;
    for (int i = 0; i < show; ++i) {
        std::cout << std::setw(8) << i
                  << std::setw(12) << std::setprecision(4) << points[i].x
                  << std::setw(12) << points[i].y
                  << std::setw(10) << points[i].label
                  << std::setw(12) << result.assignments[i]
                  << std::endl;
    }
}

int main() {
    try {
        int n, k, m;
        std::cout << "Число точек (n): ";
        std::cin >> n;
        if (n <= 0) throw std::invalid_argument("n должно быть > 0");

        std::cout << "Число классов (k): ";
        std::cin >> k;
        if (k <= 0) throw std::invalid_argument("k должно быть > 0");

        std::cout << "Число кластеров (m): ";
        std::cin >> m;
        if (m <= 0) throw std::invalid_argument("m должно быть > 0");

        std::string input_mode;
        std::cout << "Ввод данных (gen/manual): ";
        std::cin >> input_mode;

        std::vector<Point> points;

        if (input_mode == "gen") {
            std::cout << "\nГенерация " << n << " точек..." << std::endl;
            points = generate_points(n, k);
        } else {
            std::cout << "Введите " << n << " троек (x y class):" << std::endl;
            points.resize(n);
            for (int i = 0; i < n; ++i) {
                std::cin >> points[i].x >> points[i].y >> points[i].label;
            }
        }

        std::cout << "Кластеризация..." << std::endl;
        auto result = kmeans(points, m);

        print_results(result, points);

    } catch (const std::exception& e) {
        std::cerr << "Ошибка: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
