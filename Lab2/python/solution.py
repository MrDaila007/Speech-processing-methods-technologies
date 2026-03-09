import math
import random
from collections import Counter


def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def find_nearest_centroid(point, centroids):
    best = 0
    best_dist = float("inf")
    for i, c in enumerate(centroids):
        d = euclidean_distance(point[0], point[1], c[0], c[1])
        if d < best_dist:
            best_dist = d
            best = i
    return best


def init_centroids(points, m, seed=42):
    if len(points) < m:
        raise ValueError("Число точек меньше числа кластеров")
    rng = random.Random(seed)
    indices = rng.sample(range(len(points)), m)
    return [(points[i][0], points[i][1]) for i in indices]


def assign_clusters(points, centroids):
    return [find_nearest_centroid(p, centroids) for p in points]


def update_centroids(points, assignments, m):
    sums = [[0.0, 0.0, 0] for _ in range(m)]
    for i, p in enumerate(points):
        c = assignments[i]
        sums[c][0] += p[0]
        sums[c][1] += p[1]
        sums[c][2] += 1

    centroids = []
    for s in sums:
        if s[2] > 0:
            centroids.append((s[0] / s[2], s[1] / s[2]))
        else:
            centroids.append((0.0, 0.0))
    return centroids


def determine_cluster_class(points, assignments, cluster_id):
    labels = [p[2] for i, p in enumerate(points) if assignments[i] == cluster_id]
    if not labels:
        return -1
    counter = Counter(labels)
    return counter.most_common(1)[0][0]


def compute_total_distance(points, assignments, centroids):
    total = 0.0
    for i, p in enumerate(points):
        c = centroids[assignments[i]]
        total += euclidean_distance(p[0], p[1], c[0], c[1])
    return total


def kmeans(points, m, max_iter=300, tol=1e-6, seed=42):
    if not points:
        raise ValueError("Массив точек пуст")
    if m <= 0:
        raise ValueError("Число кластеров должно быть > 0")

    centroids = init_centroids(points, m, seed)
    assignments = []

    for iteration in range(1, max_iter + 1):
        assignments = assign_clusters(points, centroids)
        new_centroids = update_centroids(points, assignments, m)

        shift = sum(
            euclidean_distance(c[0], c[1], nc[0], nc[1])
            for c, nc in zip(centroids, new_centroids)
        )
        centroids = new_centroids

        if shift < tol:
            break

    cluster_info = []
    total_mismatched = 0

    for c in range(m):
        assigned_class = determine_cluster_class(points, assignments, c)
        total = 0
        mismatched = 0
        for i, p in enumerate(points):
            if assignments[i] == c:
                total += 1
                if p[2] != assigned_class:
                    mismatched += 1
        cluster_info.append({
            "cluster_id": c,
            "assigned_class": assigned_class,
            "total_points": total,
            "mismatched_points": mismatched,
        })
        total_mismatched += mismatched

    error_rate = total_mismatched / len(points)
    total_dist = compute_total_distance(points, assignments, centroids)

    return {
        "assignments": assignments,
        "centroids": centroids,
        "cluster_info": cluster_info,
        "error_rate": error_rate,
        "total_distance": total_dist,
        "iterations": iteration,
    }


def generate_points(n, k, seed=42):
    rng = random.Random(seed)
    centers = [(rng.uniform(-50, 50), rng.uniform(-50, 50)) for _ in range(k)]

    points = []
    for _ in range(n):
        label = rng.randint(0, k - 1)
        x = centers[label][0] + rng.gauss(0, 5)
        y = centers[label][1] + rng.gauss(0, 5)
        points.append((x, y, label))
    return points


def print_results(result, points):
    print(f"\n=== Результаты кластеризации ===")
    print(f"Итераций:            {result['iterations']}")
    print(f"Суммарное расстояние: {result['total_distance']:.2f}")
    print(f"Доля ошибок:         {result['error_rate'] * 100:.4f}%")

    print(f"\nКластеры:")
    print(f"{'Кластер':>10}{'Класс':>10}{'Точек':>10}{'Ошибок':>12}{'Центр X':>14}{'Центр Y':>14}")
    print("-" * 70)
    for ci in result["cluster_info"]:
        cid = ci["cluster_id"]
        cx, cy = result["centroids"][cid]
        print(f"{cid:>10}{ci['assigned_class']:>10}{ci['total_points']:>10}"
              f"{ci['mismatched_points']:>12}{cx:>14.4f}{cy:>14.4f}")

    show = min(len(points), 20)
    print(f"\nПервые {show} точек:")
    print(f"{'#':>8}{'X':>12}{'Y':>12}{'Класс':>10}{'Кластер':>12}")
    print("-" * 54)
    for i in range(show):
        p = points[i]
        print(f"{i:>8}{p[0]:>12.4f}{p[1]:>12.4f}{p[2]:>10}{result['assignments'][i]:>12}")


def main():
    n = int(input("Число точек (n): "))
    k = int(input("Число классов (k): "))
    m = int(input("Число кластеров (m): "))
    mode = input("Ввод данных (gen/manual): ").strip()

    if mode == "gen":
        print(f"\nГенерация {n} точек...")
        points = generate_points(n, k)
    else:
        print(f"Введите {n} троек (x y class):")
        points = []
        for _ in range(n):
            parts = input().split()
            points.append((float(parts[0]), float(parts[1]), int(parts[2])))

    print("Кластеризация...")
    result = kmeans(points, m)
    print_results(result, points)


if __name__ == "__main__":
    main()
