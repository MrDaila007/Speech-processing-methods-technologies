FREQ_MIN <- 20.0
FREQ_MAX <- 20000.0
VALID_RATES <- c(48, 96, 128, 176, 256)

validate_params <- function(n_seconds, bitrate, sample_rate_khz) {
  if (n_seconds <= 0) stop("n должно быть > 0")
  if (bitrate <= 0 || bitrate > 24) stop("Битрейт должен быть от 1 до 24")
  if (!(sample_rate_khz %in% VALID_RATES)) stop("Недопустимая частота записи")
  list(n_seconds = n_seconds, bitrate = bitrate, sample_rate_khz = sample_rate_khz)
}

generate_signal <- function(count, seed = 42) {
  set.seed(seed)
  runif(count, min = FREQ_MIN, max = FREQ_MAX)
}

compute_step <- function(bitrate) {
  levels <- bitwShiftL(1L, bitrate) - 1
  (FREQ_MAX - FREQ_MIN) / levels
}

quantize <- function(value, step) {
  relative <- value - FREQ_MIN
  level <- round(relative / step)
  FREQ_MIN + level * step
}

discretize <- function(signal, n_seconds, bitrate, sample_rate_khz) {
  total_input <- length(signal)
  total_samples <- as.integer(sample_rate_khz) * 1000L * n_seconds
  step <- compute_step(bitrate)
  samples_per_bin <- total_input / total_samples

  output <- numeric(total_samples)
  for (i in seq_len(total_samples)) {
    start <- as.integer((i - 1) * samples_per_bin) + 1L
    end <- as.integer(i * samples_per_bin)
    if (end > total_input) end <- total_input
    if (start > end) start <- end

    avg <- mean(signal[start:end])
    output[i] <- quantize(avg, step)
  }
  output
}

print_results <- function(output, bitrate, sample_rate_khz, n_seconds) {
  levels <- bitwShiftL(1L, bitrate) - 1
  step <- compute_step(bitrate)
  total_samples <- as.integer(sample_rate_khz) * 1000L * n_seconds

  cat("\n=== Результаты дискретизации ===\n")
  cat(sprintf("Частота записи:   %d KHz\n", sample_rate_khz))
  cat(sprintf("Битрейт:          %d бит\n", bitrate))
  cat(sprintf("Уровней квант.:   %d\n", levels + 1))
  cat(sprintf("Шаг квантования:  %.4f Hz\n", step))
  cat(sprintf("Всего сэмплов:    %d\n", total_samples))

  show <- min(length(output), 20)
  cat(sprintf("\nПервые %d сэмплов:\n", show))
  cat(sprintf("%10s%16s\n", "Сэмпл", "Частота (Hz)"))
  cat(strrep("-", 26), "\n")
  for (i in seq_len(show)) {
    cat(sprintf("%10d%16.4f\n", i - 1, output[i]))
  }

  cat("\nСтатистика:\n")
  cat(sprintf("  Min: %.4f Hz\n", min(output)))
  cat(sprintf("  Max: %.4f Hz\n", max(output)))
  cat(sprintf("  Avg: %.4f Hz\n", mean(output)))
}

main <- function() {
  cat("Количество секунд аудио (n): ")
  n_seconds <- as.integer(readLines(con = "stdin", n = 1))

  cat("Битрейт (бит): ")
  bitrate <- as.integer(readLines(con = "stdin", n = 1))

  cat("Частота записи (48, 96, 128, 176, 256) KHz: ")
  sample_rate_khz <- as.integer(readLines(con = "stdin", n = 1))

  params <- validate_params(n_seconds, bitrate, sample_rate_khz)

  total_values <- 1000000L * params$n_seconds
  cat(sprintf("\nГенерация %d значений сигнала...\n", total_values))
  signal <- generate_signal(total_values)

  cat("Дискретизация...\n")
  output <- discretize(signal, params$n_seconds, params$bitrate, params$sample_rate_khz)

  print_results(output, params$bitrate, params$sample_rate_khz, params$n_seconds)
}

if (!interactive() && identical(sys.frame(1)$ofile, NULL) ||
    (!interactive() && !exists(".IS_TEST"))) {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) == 0 || args[1] != "--test") {
    main()
  }
}
