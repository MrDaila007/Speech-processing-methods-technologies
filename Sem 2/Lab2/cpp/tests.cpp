#include <iostream>
#include <cmath>
#include <string>
#include "kmeans.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT_TRUE(expr, msg) \
    if (!(expr)) { \
        std::cerr << "  FAIL: " << msg << std::endl; \
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
          std::cerr << "  FAIL: " << msg << std::endl; \
          tests_failed++; \
      } else { \
          tests_passed++; \
      } \
    }

// ========== Unit: distance ==========

void test_distance_zero() {
    std::cout << "[distance] same point = 0..." << std::endl;
    ASSERT_NEAR(distance(3.0, 4.0, 3.0, 4.0), 0.0, 1e-10, "zero distance");
}

void test_distance_known() {
    std::cout << "[distance] known triangle..." << std::endl;
    ASSERT_NEAR(distance(0.0, 0.0, 3.0, 4.0), 5.0, 1e-10, "3-4-5 triangle");
}

void test_distance_symmetric() {
    std::cout << "[distance] symmetry..." << std::endl;
    double d1 = distance(1.0, 2.0, 4.0, 6.0);
    double d2 = distance(4.0, 6.0, 1.0, 2.0);
    ASSERT_NEAR(d1, d2, 1e-10, "d(a,b) == d(b,a)");
}

// ========== Unit: find_nearest_centroid ==========

void test_nearest_centroid() {
    std::cout << "[find_nearest] picks closest..." << std::endl;
    std::vector<Centroid> centroids = {{0, 0}, {10, 10}, {20, 20}};
    Point p = {9.0, 11.0, 0};
    ASSERT_TRUE(find_nearest_centroid(p, centroids) == 1, "nearest is centroid 1");
}

void test_nearest_centroid_exact() {
    std::cout << "[find_nearest] exact match..." << std::endl;
    std::vector<Centroid> centroids = {{5, 5}, {10, 10}};
    Point p = {5.0, 5.0, 0};
    ASSERT_TRUE(find_nearest_centroid(p, centroids) == 0, "exact match centroid 0");
}

// ========== Unit: init_centroids ==========

void test_init_centroids_count() {
    std::cout << "[init_centroids] correct count..." << std::endl;
    std::vector<Point> points = {{0,0,0}, {1,1,0}, {2,2,0}, {3,3,0}, {4,4,0}};
    auto c = init_centroids(points, 3);
    ASSERT_TRUE(c.size() == 3, "3 centroids initialized");
}

void test_init_centroids_from_points() {
    std::cout << "[init_centroids] values from input points..." << std::endl;
    std::vector<Point> points = {{0,0,0}, {10,10,1}, {20,20,2}};
    auto c = init_centroids(points, 2);
    // Each centroid must match some point
    for (const auto& ci : c) {
        bool found = false;
        for (const auto& p : points) {
            if (std::abs(ci.x - p.x) < 1e-10 && std::abs(ci.y - p.y) < 1e-10) {
                found = true;
                break;
            }
        }
        ASSERT_TRUE(found, "centroid matches an input point");
    }
}

void test_init_centroids_too_few_points() {
    std::cout << "[init_centroids] throws if n < m..." << std::endl;
    std::vector<Point> points = {{0,0,0}};
    ASSERT_THROWS(init_centroids(points, 5), "too few points throws");
}

// ========== Unit: assign_clusters ==========

void test_assign_clusters() {
    std::cout << "[assign_clusters] correct assignment..." << std::endl;
    std::vector<Point> points = {{0,0,0}, {10,10,1}, {20,20,2}};
    std::vector<Centroid> centroids = {{1,1}, {19,19}};
    auto a = assign_clusters(points, centroids);
    ASSERT_TRUE(a[0] == 0, "point(0,0) -> centroid(1,1)");
    ASSERT_TRUE(a[1] == 0 || a[1] == 1, "point(10,10) assigned");
    ASSERT_TRUE(a[2] == 1, "point(20,20) -> centroid(19,19)");
}

// ========== Unit: update_centroids ==========

void test_update_centroids() {
    std::cout << "[update_centroids] recomputes mean..." << std::endl;
    std::vector<Point> points = {{0,0,0}, {2,2,0}, {10,10,1}, {12,12,1}};
    std::vector<int> assignments = {0, 0, 1, 1};
    auto c = update_centroids(points, assignments, 2);
    ASSERT_NEAR(c[0].x, 1.0, 1e-10, "cluster 0 center x");
    ASSERT_NEAR(c[0].y, 1.0, 1e-10, "cluster 0 center y");
    ASSERT_NEAR(c[1].x, 11.0, 1e-10, "cluster 1 center x");
    ASSERT_NEAR(c[1].y, 11.0, 1e-10, "cluster 1 center y");
}

// ========== Unit: determine_cluster_class ==========

void test_determine_cluster_class_majority() {
    std::cout << "[determine_class] majority vote..." << std::endl;
    std::vector<Point> points = {{0,0,0}, {1,1,0}, {2,2,1}, {3,3,0}};
    std::vector<int> assignments = {0, 0, 0, 0};
    int cls = determine_cluster_class(points, assignments, 0);
    ASSERT_TRUE(cls == 0, "majority class is 0");
}

void test_determine_cluster_class_single() {
    std::cout << "[determine_class] single point..." << std::endl;
    std::vector<Point> points = {{5,5,3}};
    std::vector<int> assignments = {0};
    int cls = determine_cluster_class(points, assignments, 0);
    ASSERT_TRUE(cls == 3, "single point class is 3");
}

// ========== Unit: compute_total_distance ==========

void test_total_distance() {
    std::cout << "[total_distance] known values..." << std::endl;
    std::vector<Point> points = {{0,0,0}, {3,4,0}};
    std::vector<Centroid> centroids = {{0,0}};
    std::vector<int> assignments = {0, 0};
    double d = compute_total_distance(points, assignments, centroids);
    ASSERT_NEAR(d, 5.0, 1e-10, "0 + 5 = 5");
}

// ========== Unit: generate_points ==========

void test_generate_points_count() {
    std::cout << "[generate_points] correct count..." << std::endl;
    auto pts = generate_points(100, 3);
    ASSERT_TRUE(pts.size() == 100, "100 points");
}

void test_generate_points_labels() {
    std::cout << "[generate_points] labels in range..." << std::endl;
    auto pts = generate_points(500, 4);
    for (const auto& p : pts) {
        ASSERT_TRUE(p.label >= 0 && p.label < 4, "label in [0,4)");
    }
}

void test_generate_points_deterministic() {
    std::cout << "[generate_points] deterministic..." << std::endl;
    auto a = generate_points(50, 3, 99);
    auto b = generate_points(50, 3, 99);
    bool same = true;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].x != b[i].x || a[i].y != b[i].y || a[i].label != b[i].label) {
            same = false;
            break;
        }
    }
    ASSERT_TRUE(same, "same seed -> same points");
}

// ========== Integration: kmeans ==========

void test_kmeans_cluster_count() {
    std::cout << "[kmeans] correct number of clusters..." << std::endl;
    auto pts = generate_points(200, 3);
    auto result = kmeans(pts, 3);
    ASSERT_TRUE(result.centroids.size() == 3, "3 centroids");
    ASSERT_TRUE(result.cluster_info.size() == 3, "3 cluster infos");
}

void test_kmeans_assignments_valid() {
    std::cout << "[kmeans] assignments in [0, m)..." << std::endl;
    auto pts = generate_points(200, 3);
    auto result = kmeans(pts, 5);
    for (int a : result.assignments) {
        ASSERT_TRUE(a >= 0 && a < 5, "assignment in range");
    }
}

void test_kmeans_all_points_assigned() {
    std::cout << "[kmeans] all points assigned..." << std::endl;
    auto pts = generate_points(150, 2);
    auto result = kmeans(pts, 4);
    ASSERT_TRUE(result.assignments.size() == 150, "150 assignments");
}

void test_kmeans_error_rate_range() {
    std::cout << "[kmeans] error rate in [0, 1]..." << std::endl;
    auto pts = generate_points(300, 3);
    auto result = kmeans(pts, 3);
    ASSERT_TRUE(result.error_rate >= 0.0 && result.error_rate <= 1.0, "error in [0,1]");
}

void test_kmeans_well_separated_clusters() {
    std::cout << "[kmeans] well-separated clusters -> low error..." << std::endl;
    // Create 3 tight clusters far apart
    std::vector<Point> pts;
    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 0.5);
    for (int i = 0; i < 100; ++i) {
        pts.push_back({0.0 + noise(rng), 0.0 + noise(rng), 0});
        pts.push_back({100.0 + noise(rng), 0.0 + noise(rng), 1});
        pts.push_back({0.0 + noise(rng), 100.0 + noise(rng), 2});
    }
    // Try multiple seeds, pick best (mimics k-means++ restarts)
    double best_error = 1.0;
    for (unsigned s = 0; s < 10; ++s) {
        auto result = kmeans(pts, 3, 300, 1e-6, s);
        if (result.error_rate < best_error) best_error = result.error_rate;
    }
    ASSERT_TRUE(best_error < 0.05, "well-separated -> error < 5%");
}

void test_kmeans_single_cluster() {
    std::cout << "[kmeans] m=1 puts all in one cluster..." << std::endl;
    auto pts = generate_points(100, 3);
    auto result = kmeans(pts, 1);
    for (int a : result.assignments) {
        ASSERT_TRUE(a == 0, "all in cluster 0");
    }
}

void test_kmeans_m_equals_n() {
    std::cout << "[kmeans] m=n gives one point per cluster..." << std::endl;
    std::vector<Point> pts = {{0,0,0}, {10,10,1}, {20,20,2}};
    auto result = kmeans(pts, 3);
    // Each point should be in a different cluster
    std::vector<bool> used(3, false);
    for (int a : result.assignments) {
        used[a] = true;
    }
    int distinct = 0;
    for (bool u : used) if (u) distinct++;
    ASSERT_TRUE(distinct == 3, "3 distinct clusters");
}

void test_kmeans_converges() {
    std::cout << "[kmeans] converges within max_iter..." << std::endl;
    auto pts = generate_points(200, 3);
    auto result = kmeans(pts, 3, 300);
    ASSERT_TRUE(result.iterations <= 300, "converged within 300 iterations");
}

void test_kmeans_total_distance_positive() {
    std::cout << "[kmeans] total distance >= 0..." << std::endl;
    auto pts = generate_points(100, 2);
    auto result = kmeans(pts, 2);
    ASSERT_TRUE(result.total_distance >= 0.0, "distance >= 0");
}

void test_kmeans_empty_throws() {
    std::cout << "[kmeans] empty input throws..." << std::endl;
    std::vector<Point> empty;
    ASSERT_THROWS(kmeans(empty, 3), "empty throws");
}

void test_kmeans_zero_clusters_throws() {
    std::cout << "[kmeans] m=0 throws..." << std::endl;
    auto pts = generate_points(10, 2);
    ASSERT_THROWS(kmeans(pts, 0), "m=0 throws");
}

void test_kmeans_error_count_matches() {
    std::cout << "[kmeans] error count consistent..." << std::endl;
    auto pts = generate_points(200, 3);
    auto result = kmeans(pts, 3);
    int total_mis = 0;
    int total_pts = 0;
    for (const auto& ci : result.cluster_info) {
        total_mis += ci.mismatched_points;
        total_pts += ci.total_points;
    }
    ASSERT_TRUE(total_pts == 200, "total points == 200");
    ASSERT_NEAR(result.error_rate, static_cast<double>(total_mis) / 200.0, 1e-10,
                "error_rate matches sum");
}

// ========== Main ==========

int main() {
    std::cout << "=== Running Tests ===" << std::endl << std::endl;

    // Unit: distance
    test_distance_zero();
    test_distance_known();
    test_distance_symmetric();

    // Unit: find_nearest_centroid
    test_nearest_centroid();
    test_nearest_centroid_exact();

    // Unit: init_centroids
    test_init_centroids_count();
    test_init_centroids_from_points();
    test_init_centroids_too_few_points();

    // Unit: assign_clusters
    test_assign_clusters();

    // Unit: update_centroids
    test_update_centroids();

    // Unit: determine_cluster_class
    test_determine_cluster_class_majority();
    test_determine_cluster_class_single();

    // Unit: compute_total_distance
    test_total_distance();

    // Unit: generate_points
    test_generate_points_count();
    test_generate_points_labels();
    test_generate_points_deterministic();

    // Integration: kmeans
    test_kmeans_cluster_count();
    test_kmeans_assignments_valid();
    test_kmeans_all_points_assigned();
    test_kmeans_error_rate_range();
    test_kmeans_well_separated_clusters();
    test_kmeans_single_cluster();
    test_kmeans_m_equals_n();
    test_kmeans_converges();
    test_kmeans_total_distance_positive();
    test_kmeans_empty_throws();
    test_kmeans_zero_clusters_throws();
    test_kmeans_error_count_matches();

    std::cout << std::endl << "=== Results ===" << std::endl;
    std::cout << "Passed: " << tests_passed << std::endl;
    std::cout << "Failed: " << tests_failed << std::endl;

    return tests_failed > 0 ? 1 : 0;
}
