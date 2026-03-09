euclidean_distance <- function(x1, y1, x2, y2) {
  sqrt((x1 - x2)^2 + (y1 - y2)^2)
}

find_nearest_centroid <- function(px, py, centroids) {
  dists <- mapply(function(cx, cy) euclidean_distance(px, py, cx, cy),
                  centroids$x, centroids$y)
  which.min(dists)
}

init_centroids <- function(points, m, seed = 42) {
  if (nrow(points) < m) stop("Число точек меньше числа кластеров")
  set.seed(seed)
  idx <- sample(nrow(points), m)
  data.frame(x = points$x[idx], y = points$y[idx])
}

assign_clusters <- function(points, centroids) {
  sapply(seq_len(nrow(points)), function(i) {
    find_nearest_centroid(points$x[i], points$y[i], centroids)
  })
}

update_centroids <- function(points, assignments, m) {
  centroids <- data.frame(x = numeric(m), y = numeric(m))
  for (c in seq_len(m)) {
    mask <- assignments == c
    if (any(mask)) {
      centroids$x[c] <- mean(points$x[mask])
      centroids$y[c] <- mean(points$y[mask])
    }
  }
  centroids
}

determine_cluster_class <- function(points, assignments, cluster_id) {
  labels <- points$label[assignments == cluster_id]
  if (length(labels) == 0) return(-1L)
  tbl <- table(labels)
  as.integer(names(which.max(tbl)))
}

compute_total_distance <- function(points, assignments, centroids) {
  total <- 0.0
  for (i in seq_len(nrow(points))) {
    c <- assignments[i]
    total <- total + euclidean_distance(points$x[i], points$y[i],
                                        centroids$x[c], centroids$y[c])
  }
  total
}

kmeans_custom <- function(points, m, max_iter = 300, tol = 1e-6, seed = 42) {
  if (nrow(points) == 0) stop("Массив точек пуст")
  if (m <= 0) stop("Число кластеров должно быть > 0")

  centroids <- init_centroids(points, m, seed)
  assignments <- integer(nrow(points))

  iter <- 0
  for (it in seq_len(max_iter)) {
    iter <- it
    assignments <- assign_clusters(points, centroids)
    new_centroids <- update_centroids(points, assignments, m)

    shift <- sum(mapply(function(i) {
      euclidean_distance(centroids$x[i], centroids$y[i],
                         new_centroids$x[i], new_centroids$y[i])
    }, seq_len(m)))

    centroids <- new_centroids
    if (shift < tol) break
  }

  cluster_info <- list()
  total_mismatched <- 0L

  for (c in seq_len(m)) {
    assigned_class <- determine_cluster_class(points, assignments, c)
    mask <- assignments == c
    total_pts <- sum(mask)
    mismatched <- sum(points$label[mask] != assigned_class)

    cluster_info[[c]] <- list(
      cluster_id = c,
      assigned_class = assigned_class,
      total_points = total_pts,
      mismatched_points = mismatched
    )
    total_mismatched <- total_mismatched + mismatched
  }

  error_rate <- total_mismatched / nrow(points)
  total_dist <- compute_total_distance(points, assignments, centroids)

  list(
    assignments = assignments,
    centroids = centroids,
    cluster_info = cluster_info,
    error_rate = error_rate,
    total_distance = total_dist,
    iterations = iter
  )
}

generate_points <- function(n, k, seed = 42) {
  set.seed(seed)
  centers_x <- runif(k, -50, 50)
  centers_y <- runif(k, -50, 50)

  labels <- sample(0:(k - 1), n, replace = TRUE)
  x <- centers_x[labels + 1] + rnorm(n, 0, 5)
  y <- centers_y[labels + 1] + rnorm(n, 0, 5)

  data.frame(x = x, y = y, label = labels)
}

print_results <- function(result, points) {
  cat("\n=== Результаты кластеризации ===\n")
  cat(sprintf("Итераций:            %d\n", result$iterations))
  cat(sprintf("Суммарное расстояние: %.2f\n", result$total_distance))
  cat(sprintf("Доля ошибок:         %.4f%%\n", result$error_rate * 100))

  cat("\nКластеры:\n")
  cat(sprintf("%10s%10s%10s%12s%14s%14s\n",
              "Кластер", "Класс", "Точек", "Ошибок", "Центр X", "Центр Y"))
  cat(strrep("-", 70), "\n")
  for (ci in result$cluster_info) {
    cat(sprintf("%10d%10d%10d%12d%14.4f%14.4f\n",
                ci$cluster_id, ci$assigned_class, ci$total_points,
                ci$mismatched_points,
                result$centroids$x[ci$cluster_id],
                result$centroids$y[ci$cluster_id]))
  }

  show <- min(nrow(points), 20)
  cat(sprintf("\nПервые %d точек:\n", show))
  cat(sprintf("%8s%12s%12s%10s%12s\n", "#", "X", "Y", "Класс", "Кластер"))
  cat(strrep("-", 54), "\n")
  for (i in seq_len(show)) {
    cat(sprintf("%8d%12.4f%12.4f%10d%12d\n",
                i - 1, points$x[i], points$y[i], points$label[i],
                result$assignments[i]))
  }
}

main <- function() {
  cat("Число точек (n): ")
  n <- as.integer(readLines(con = "stdin", n = 1))

  cat("Число классов (k): ")
  k <- as.integer(readLines(con = "stdin", n = 1))

  cat("Число кластеров (m): ")
  m <- as.integer(readLines(con = "stdin", n = 1))

  cat("Ввод данных (gen/manual): ")
  mode <- trimws(readLines(con = "stdin", n = 1))

  if (mode == "gen") {
    cat(sprintf("\nГенерация %d точек...\n", n))
    points <- generate_points(n, k)
  } else {
    cat(sprintf("Введите %d троек (x y class):\n", n))
    x <- numeric(n); y <- numeric(n); label <- integer(n)
    for (i in seq_len(n)) {
      parts <- as.numeric(strsplit(readLines(con = "stdin", n = 1), " ")[[1]])
      x[i] <- parts[1]; y[i] <- parts[2]; label[i] <- as.integer(parts[3])
    }
    points <- data.frame(x = x, y = y, label = label)
  }

  cat("Кластеризация...\n")
  result <- kmeans_custom(points, m)
  print_results(result, points)
}

if (!interactive() && !exists(".IS_TEST")) {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) == 0 || args[1] != "--test") {
    main()
  }
}
