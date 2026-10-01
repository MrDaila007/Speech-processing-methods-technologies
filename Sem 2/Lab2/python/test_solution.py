import unittest
import math
import random
from solution import (
    euclidean_distance, find_nearest_centroid, init_centroids,
    assign_clusters, update_centroids, determine_cluster_class,
    compute_total_distance, kmeans, generate_points,
)


class TestDistance(unittest.TestCase):
    def test_zero(self):
        self.assertAlmostEqual(euclidean_distance(3, 4, 3, 4), 0.0)

    def test_known(self):
        self.assertAlmostEqual(euclidean_distance(0, 0, 3, 4), 5.0)

    def test_symmetric(self):
        d1 = euclidean_distance(1, 2, 4, 6)
        d2 = euclidean_distance(4, 6, 1, 2)
        self.assertAlmostEqual(d1, d2)


class TestFindNearest(unittest.TestCase):
    def test_picks_closest(self):
        centroids = [(0, 0), (10, 10), (20, 20)]
        self.assertEqual(find_nearest_centroid((9, 11, 0), centroids), 1)

    def test_exact_match(self):
        centroids = [(5, 5), (10, 10)]
        self.assertEqual(find_nearest_centroid((5, 5, 0), centroids), 0)


class TestInitCentroids(unittest.TestCase):
    def test_count(self):
        points = [(i, i, 0) for i in range(10)]
        c = init_centroids(points, 3)
        self.assertEqual(len(c), 3)

    def test_from_points(self):
        points = [(0, 0, 0), (10, 10, 1), (20, 20, 2)]
        c = init_centroids(points, 2)
        point_coords = {(p[0], p[1]) for p in points}
        for ci in c:
            self.assertIn(ci, point_coords)

    def test_too_few_throws(self):
        with self.assertRaises(ValueError):
            init_centroids([(0, 0, 0)], 5)


class TestAssignClusters(unittest.TestCase):
    def test_assignment(self):
        points = [(0, 0, 0), (10, 10, 1), (20, 20, 2)]
        centroids = [(1, 1), (19, 19)]
        a = assign_clusters(points, centroids)
        self.assertEqual(a[0], 0)
        self.assertEqual(a[2], 1)


class TestUpdateCentroids(unittest.TestCase):
    def test_recomputes_mean(self):
        points = [(0, 0, 0), (2, 2, 0), (10, 10, 1), (12, 12, 1)]
        assignments = [0, 0, 1, 1]
        c = update_centroids(points, assignments, 2)
        self.assertAlmostEqual(c[0][0], 1.0)
        self.assertAlmostEqual(c[0][1], 1.0)
        self.assertAlmostEqual(c[1][0], 11.0)
        self.assertAlmostEqual(c[1][1], 11.0)


class TestDetermineClass(unittest.TestCase):
    def test_majority(self):
        points = [(0, 0, 0), (1, 1, 0), (2, 2, 1), (3, 3, 0)]
        assignments = [0, 0, 0, 0]
        self.assertEqual(determine_cluster_class(points, assignments, 0), 0)

    def test_single(self):
        points = [(5, 5, 3)]
        assignments = [0]
        self.assertEqual(determine_cluster_class(points, assignments, 0), 3)


class TestTotalDistance(unittest.TestCase):
    def test_known(self):
        points = [(0, 0, 0), (3, 4, 0)]
        centroids = [(0, 0)]
        assignments = [0, 0]
        self.assertAlmostEqual(compute_total_distance(points, assignments, centroids), 5.0)


class TestGeneratePoints(unittest.TestCase):
    def test_count(self):
        pts = generate_points(100, 3)
        self.assertEqual(len(pts), 100)

    def test_labels_in_range(self):
        pts = generate_points(500, 4)
        for p in pts:
            self.assertGreaterEqual(p[2], 0)
            self.assertLess(p[2], 4)

    def test_deterministic(self):
        a = generate_points(50, 3, seed=99)
        b = generate_points(50, 3, seed=99)
        self.assertEqual(a, b)


class TestKMeans(unittest.TestCase):
    def test_cluster_count(self):
        pts = generate_points(200, 3)
        r = kmeans(pts, 3)
        self.assertEqual(len(r["centroids"]), 3)
        self.assertEqual(len(r["cluster_info"]), 3)

    def test_assignments_valid(self):
        pts = generate_points(200, 3)
        r = kmeans(pts, 5)
        for a in r["assignments"]:
            self.assertGreaterEqual(a, 0)
            self.assertLess(a, 5)

    def test_all_points_assigned(self):
        pts = generate_points(150, 2)
        r = kmeans(pts, 4)
        self.assertEqual(len(r["assignments"]), 150)

    def test_error_rate_range(self):
        pts = generate_points(300, 3)
        r = kmeans(pts, 3)
        self.assertGreaterEqual(r["error_rate"], 0.0)
        self.assertLessEqual(r["error_rate"], 1.0)

    def test_well_separated_low_error(self):
        rng = random.Random(42)
        pts = []
        for _ in range(100):
            pts.append((rng.gauss(0, 0.5), rng.gauss(0, 0.5), 0))
            pts.append((rng.gauss(100, 0.5), rng.gauss(0, 0.5), 1))
            pts.append((rng.gauss(0, 0.5), rng.gauss(100, 0.5), 2))
        r = kmeans(pts, 3)
        self.assertLess(r["error_rate"], 0.05)

    def test_single_cluster(self):
        pts = generate_points(100, 3)
        r = kmeans(pts, 1)
        for a in r["assignments"]:
            self.assertEqual(a, 0)

    def test_m_equals_n(self):
        pts = [(0, 0, 0), (10, 10, 1), (20, 20, 2)]
        r = kmeans(pts, 3)
        self.assertEqual(len(set(r["assignments"])), 3)

    def test_converges(self):
        pts = generate_points(200, 3)
        r = kmeans(pts, 3, max_iter=300)
        self.assertLessEqual(r["iterations"], 300)

    def test_total_distance_positive(self):
        pts = generate_points(100, 2)
        r = kmeans(pts, 2)
        self.assertGreaterEqual(r["total_distance"], 0.0)

    def test_empty_throws(self):
        with self.assertRaises(ValueError):
            kmeans([], 3)

    def test_zero_clusters_throws(self):
        pts = generate_points(10, 2)
        with self.assertRaises(ValueError):
            kmeans(pts, 0)

    def test_error_count_consistent(self):
        pts = generate_points(200, 3)
        r = kmeans(pts, 3)
        total_mis = sum(ci["mismatched_points"] for ci in r["cluster_info"])
        total_pts = sum(ci["total_points"] for ci in r["cluster_info"])
        self.assertEqual(total_pts, 200)
        self.assertAlmostEqual(r["error_rate"], total_mis / 200.0)


if __name__ == "__main__":
    unittest.main()
