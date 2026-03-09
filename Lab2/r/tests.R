.IS_TEST <- TRUE
source("solution.R")

tests_passed <- 0L
tests_failed <- 0L

assert_true <- function(expr, msg) {
  if (isTRUE(expr)) {
    tests_passed <<- tests_passed + 1L
  } else {
    cat(sprintf("  FAIL: %s\n", msg))
    tests_failed <<- tests_failed + 1L
  }
}

assert_near <- function(a, b, eps, msg) {
  if (abs(a - b) <= eps) {
    tests_passed <<- tests_passed + 1L
  } else {
    cat(sprintf("  FAIL: %s (expected %.6f, got %.6f)\n", msg, b, a))
    tests_failed <<- tests_failed + 1L
  }
}

assert_throws <- function(expr_fn, msg) {
  threw <- tryCatch({ expr_fn(); FALSE }, error = function(e) TRUE)
  if (threw) {
    tests_passed <<- tests_passed + 1L
  } else {
    cat(sprintf("  FAIL: %s (no error thrown)\n", msg))
    tests_failed <<- tests_failed + 1L
  }
}

# ========== distance ==========

cat("[distance] basic tests...\n")
assert_near(euclidean_distance(3, 4, 3, 4), 0.0, 1e-10, "same point = 0")
assert_near(euclidean_distance(0, 0, 3, 4), 5.0, 1e-10, "3-4-5 triangle")
d1 <- euclidean_distance(1, 2, 4, 6)
d2 <- euclidean_distance(4, 6, 1, 2)
assert_near(d1, d2, 1e-10, "symmetry")

# ========== find_nearest_centroid ==========

cat("[find_nearest] picks closest...\n")
cents <- data.frame(x = c(0, 10, 20), y = c(0, 10, 20))
assert_true(find_nearest_centroid(9, 11, cents) == 2, "nearest is centroid 2 (1-indexed)")
assert_true(find_nearest_centroid(5, 5, data.frame(x = c(5, 10), y = c(5, 10))) == 1,
            "exact match")

# ========== init_centroids ==========

cat("[init_centroids] correct count...\n")
pts5 <- data.frame(x = 0:4, y = 0:4, label = rep(0, 5))
c3 <- init_centroids(pts5, 3)
assert_true(nrow(c3) == 3, "3 centroids")

cat("[init_centroids] values from points...\n")
pts3 <- data.frame(x = c(0, 10, 20), y = c(0, 10, 20), label = 0:2)
c2 <- init_centroids(pts3, 2)
for (i in seq_len(nrow(c2))) {
  found <- any(abs(pts3$x - c2$x[i]) < 1e-10 & abs(pts3$y - c2$y[i]) < 1e-10)
  assert_true(found, paste("centroid", i, "from input"))
}

cat("[init_centroids] throws if n < m...\n")
assert_throws(function() init_centroids(data.frame(x = 0, y = 0, label = 0), 5),
              "too few points")

# ========== assign_clusters ==========

cat("[assign_clusters] correct...\n")
pts_a <- data.frame(x = c(0, 10, 20), y = c(0, 10, 20), label = 0:2)
cents_a <- data.frame(x = c(1, 19), y = c(1, 19))
a <- assign_clusters(pts_a, cents_a)
assert_true(a[1] == 1, "point(0,0) -> centroid 1")
assert_true(a[3] == 2, "point(20,20) -> centroid 2")

# ========== update_centroids ==========

cat("[update_centroids] recomputes mean...\n")
pts_u <- data.frame(x = c(0, 2, 10, 12), y = c(0, 2, 10, 12), label = c(0, 0, 1, 1))
asgn <- c(1, 1, 2, 2)
cu <- update_centroids(pts_u, asgn, 2)
assert_near(cu$x[1], 1.0, 1e-10, "cluster 1 x")
assert_near(cu$y[1], 1.0, 1e-10, "cluster 1 y")
assert_near(cu$x[2], 11.0, 1e-10, "cluster 2 x")
assert_near(cu$y[2], 11.0, 1e-10, "cluster 2 y")

# ========== determine_cluster_class ==========

cat("[determine_class] majority vote...\n")
pts_d <- data.frame(x = 0:3, y = 0:3, label = c(0, 0, 1, 0))
asgn_d <- c(1, 1, 1, 1)
assert_true(determine_cluster_class(pts_d, asgn_d, 1) == 0, "majority is 0")

cat("[determine_class] single point...\n")
pts_s <- data.frame(x = 5, y = 5, label = 3)
assert_true(determine_cluster_class(pts_s, c(1), 1) == 3, "single point class 3")

# ========== compute_total_distance ==========

cat("[total_distance] known...\n")
pts_t <- data.frame(x = c(0, 3), y = c(0, 4), label = c(0, 0))
cents_t <- data.frame(x = 0, y = 0)
assert_near(compute_total_distance(pts_t, c(1, 1), cents_t), 5.0, 1e-10, "0 + 5 = 5")

# ========== generate_points ==========

cat("[generate_points] correct count...\n")
gp <- generate_points(100, 3)
assert_true(nrow(gp) == 100, "100 points")

cat("[generate_points] labels in range...\n")
gp500 <- generate_points(500, 4)
assert_true(all(gp500$label >= 0 & gp500$label < 4), "labels in [0,4)")

cat("[generate_points] deterministic...\n")
ga <- generate_points(50, 3, seed = 99)
gb <- generate_points(50, 3, seed = 99)
assert_true(identical(ga, gb), "same seed -> same points")

# ========== kmeans_custom ==========

cat("[kmeans] cluster count...\n")
kpts <- generate_points(200, 3)
kr <- kmeans_custom(kpts, 3)
assert_true(nrow(kr$centroids) == 3, "3 centroids")
assert_true(length(kr$cluster_info) == 3, "3 cluster infos")

cat("[kmeans] assignments valid...\n")
kr5 <- kmeans_custom(generate_points(200, 3), 5)
assert_true(all(kr5$assignments >= 1 & kr5$assignments <= 5), "assignments in [1,5]")

cat("[kmeans] all points assigned...\n")
kr4 <- kmeans_custom(generate_points(150, 2), 4)
assert_true(length(kr4$assignments) == 150, "150 assignments")

cat("[kmeans] error rate in [0,1]...\n")
kr_e <- kmeans_custom(generate_points(300, 3), 3)
assert_true(kr_e$error_rate >= 0.0 && kr_e$error_rate <= 1.0, "error in [0,1]")

cat("[kmeans] well-separated -> low error...\n")
set.seed(42)
ws_x <- c(rnorm(100, 0, 0.5), rnorm(100, 100, 0.5), rnorm(100, 0, 0.5))
ws_y <- c(rnorm(100, 0, 0.5), rnorm(100, 0, 0.5), rnorm(100, 100, 0.5))
ws_l <- c(rep(0, 100), rep(1, 100), rep(2, 100))
ws_pts <- data.frame(x = ws_x, y = ws_y, label = ws_l)
best_err <- 1.0
for (s in 0:9) {
  r <- kmeans_custom(ws_pts, 3, seed = s)
  if (r$error_rate < best_err) best_err <- r$error_rate
}
assert_true(best_err < 0.05, "well-separated error < 5%")

cat("[kmeans] m=1 all in one cluster...\n")
kr1 <- kmeans_custom(generate_points(100, 3), 1)
assert_true(all(kr1$assignments == 1), "all in cluster 1")

cat("[kmeans] converges...\n")
kr_c <- kmeans_custom(generate_points(200, 3), 3, max_iter = 300)
assert_true(kr_c$iterations <= 300, "converged within 300")

cat("[kmeans] total distance >= 0...\n")
kr_d <- kmeans_custom(generate_points(100, 2), 2)
assert_true(kr_d$total_distance >= 0.0, "distance >= 0")

cat("[kmeans] empty throws...\n")
assert_throws(function() kmeans_custom(data.frame(x = numeric(0), y = numeric(0),
                                                  label = integer(0)), 3),
              "empty throws")

cat("[kmeans] m=0 throws...\n")
assert_throws(function() kmeans_custom(generate_points(10, 2), 0), "m=0 throws")

cat("[kmeans] error count consistent...\n")
kr_ec <- kmeans_custom(generate_points(200, 3), 3)
total_mis <- sum(sapply(kr_ec$cluster_info, function(ci) ci$mismatched_points))
total_pts <- sum(sapply(kr_ec$cluster_info, function(ci) ci$total_points))
assert_true(total_pts == 200, "total points == 200")
assert_near(kr_ec$error_rate, total_mis / 200.0, 1e-10, "error_rate matches sum")

# ========== Results ==========

cat(sprintf("\n=== Results ===\nPassed: %d\nFailed: %d\n", tests_passed, tests_failed))
if (tests_failed > 0) quit(status = 1)
