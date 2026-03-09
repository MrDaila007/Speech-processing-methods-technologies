#pragma once

#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <limits>

struct Point {
    double x;
    double y;
    int label;
};

struct Centroid {
    double x;
    double y;
};

struct ClusterInfo {
    int cluster_id;
    int assigned_class;
    int total_points;
    int mismatched_points;
};

struct KMeansResult {
    std::vector<int> assignments;
    std::vector<Centroid> centroids;
    std::vector<ClusterInfo> cluster_info;
    double error_rate;
    double total_distance;
    int iterations;
};

inline double distance(double x1, double y1, double x2, double y2) {
    double dx = x1 - x2;
    double dy = y1 - y2;
    return std::sqrt(dx * dx + dy * dy);
}

inline int find_nearest_centroid(const Point& p, const std::vector<Centroid>& centroids) {
    int best = 0;
    double best_dist = std::numeric_limits<double>::max();
    for (int i = 0; i < static_cast<int>(centroids.size()); ++i) {
        double d = distance(p.x, p.y, centroids[i].x, centroids[i].y);
        if (d < best_dist) {
            best_dist = d;
            best = i;
        }
    }
    return best;
}

inline std::vector<Centroid> init_centroids(const std::vector<Point>& points,
                                            int m, unsigned seed = 42) {
    if (static_cast<int>(points.size()) < m) {
        throw std::invalid_argument("Число точек меньше числа кластеров");
    }

    std::vector<int> indices(points.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(indices.begin(), indices.end(), rng);

    std::vector<Centroid> centroids(m);
    for (int i = 0; i < m; ++i) {
        centroids[i] = {points[indices[i]].x, points[indices[i]].y};
    }
    return centroids;
}

inline std::vector<int> assign_clusters(const std::vector<Point>& points,
                                        const std::vector<Centroid>& centroids) {
    std::vector<int> assignments(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
        assignments[i] = find_nearest_centroid(points[i], centroids);
    }
    return assignments;
}

inline std::vector<Centroid> update_centroids(const std::vector<Point>& points,
                                              const std::vector<int>& assignments,
                                              int m) {
    std::vector<Centroid> centroids(m, {0.0, 0.0});
    std::vector<int> counts(m, 0);

    for (size_t i = 0; i < points.size(); ++i) {
        int c = assignments[i];
        centroids[c].x += points[i].x;
        centroids[c].y += points[i].y;
        counts[c]++;
    }

    for (int i = 0; i < m; ++i) {
        if (counts[i] > 0) {
            centroids[i].x /= counts[i];
            centroids[i].y /= counts[i];
        }
    }
    return centroids;
}

inline int determine_cluster_class(const std::vector<Point>& points,
                                   const std::vector<int>& assignments,
                                   int cluster_id) {
    std::unordered_map<int, int> class_counts;
    for (size_t i = 0; i < points.size(); ++i) {
        if (assignments[i] == cluster_id) {
            class_counts[points[i].label]++;
        }
    }

    int best_class = -1;
    int best_count = 0;
    for (const auto& pair : class_counts) {
        if (pair.second > best_count) {
            best_count = pair.second;
            best_class = pair.first;
        }
    }
    return best_class;
}

inline double compute_total_distance(const std::vector<Point>& points,
                                     const std::vector<int>& assignments,
                                     const std::vector<Centroid>& centroids) {
    double total = 0.0;
    for (size_t i = 0; i < points.size(); ++i) {
        const auto& c = centroids[assignments[i]];
        total += distance(points[i].x, points[i].y, c.x, c.y);
    }
    return total;
}

inline KMeansResult kmeans(const std::vector<Point>& points, int m,
                           int max_iter = 300, double tol = 1e-6,
                           unsigned seed = 42) {
    if (points.empty()) {
        throw std::invalid_argument("Массив точек пуст");
    }
    if (m <= 0) {
        throw std::invalid_argument("Число кластеров должно быть > 0");
    }

    auto centroids = init_centroids(points, m, seed);
    std::vector<int> assignments;
    int iter = 0;

    for (; iter < max_iter; ++iter) {
        assignments = assign_clusters(points, centroids);
        auto new_centroids = update_centroids(points, assignments, m);

        double shift = 0.0;
        for (int i = 0; i < m; ++i) {
            shift += distance(centroids[i].x, centroids[i].y,
                              new_centroids[i].x, new_centroids[i].y);
        }

        centroids = new_centroids;

        if (shift < tol) {
            ++iter;
            break;
        }
    }

    // Determine cluster classes and error rate
    std::vector<ClusterInfo> info(m);
    int total_mismatched = 0;

    for (int c = 0; c < m; ++c) {
        int assigned_class = determine_cluster_class(points, assignments, c);
        int total = 0;
        int mismatched = 0;

        for (size_t i = 0; i < points.size(); ++i) {
            if (assignments[i] == c) {
                total++;
                if (points[i].label != assigned_class) {
                    mismatched++;
                }
            }
        }

        info[c] = {c, assigned_class, total, mismatched};
        total_mismatched += mismatched;
    }

    double error_rate = static_cast<double>(total_mismatched) / points.size();
    double total_dist = compute_total_distance(points, assignments, centroids);

    return {assignments, centroids, info, error_rate, total_dist, iter};
}

inline std::vector<Point> generate_points(int n, int k, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> offset_dist(-50.0, 50.0);
    std::normal_distribution<double> spread(0.0, 5.0);

    // Generate k class centers
    std::vector<Centroid> centers(k);
    for (int i = 0; i < k; ++i) {
        centers[i] = {offset_dist(rng), offset_dist(rng)};
    }

    std::uniform_int_distribution<int> class_dist(0, k - 1);
    std::vector<Point> points(n);
    for (int i = 0; i < n; ++i) {
        int label = class_dist(rng);
        points[i] = {
            centers[label].x + spread(rng),
            centers[label].y + spread(rng),
            label
        };
    }
    return points;
}
