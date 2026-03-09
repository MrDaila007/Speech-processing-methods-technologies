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

# ========== compute_step ==========

cat("[compute_step] various bitrates...\n")
assert_near(compute_step(1), 19980.0, 0.001, "1-bit step = 19980")
assert_near(compute_step(8), 19980.0 / 255, 0.001, "8-bit step")
assert_near(compute_step(16), 19980.0 / 65535, 1e-6, "16-bit step")

# ========== quantize ==========

cat("[quantize] exact level value...\n")
step8 <- compute_step(8)
assert_near(quantize(20.0, step8), 20.0, 0.001, "quantize(20) == 20")
assert_near(quantize(20000.0, step8), 20000.0, 0.001, "quantize(20000) == 20000")

cat("[quantize] rounds to nearest level...\n")
assert_near(quantize(59.0, step8), 20.0, 0.01, "59 rounds down to 20")
assert_near(quantize(60.0, step8), 20.0 + step8, 0.01, "60 rounds up")

cat("[quantize] midrange value...\n")
step16 <- compute_step(16)
mid_result <- quantize(10010.0, step16)
assert_true(abs(mid_result - 10010.0) <= step16, "quantized within one step")
level_f <- (mid_result - FREQ_MIN) / step16
assert_near(level_f, round(level_f), 1e-6, "result on exact level")

cat("[quantize] boundary values...\n")
assert_true(quantize(20.1, step8) >= FREQ_MIN, "low boundary >= FREQ_MIN")
assert_true(quantize(19999.9, step8) <= FREQ_MAX + 0.001, "high boundary <= FREQ_MAX")

# ========== generate_signal ==========

cat("[generate_signal] correct count...\n")
sig <- generate_signal(1000)
assert_true(length(sig) == 1000, "generated 1000 values")

cat("[generate_signal] values in range...\n")
sig10k <- generate_signal(10000)
assert_true(all(sig10k >= FREQ_MIN) && all(sig10k <= FREQ_MAX), "all in [20, 20000]")

cat("[generate_signal] deterministic...\n")
a <- generate_signal(100, seed = 123)
b <- generate_signal(100, seed = 123)
assert_true(identical(a, b), "same seed -> same signal")

cat("[generate_signal] different seeds differ...\n")
a <- generate_signal(100, seed = 1)
b <- generate_signal(100, seed = 2)
assert_true(!identical(a, b), "different seeds -> different signals")

# ========== validate_params ==========

cat("[validate_params] valid inputs...\n")
p <- validate_params(1, 8, 48)
assert_true(p$n_seconds == 1 && p$bitrate == 8 && p$sample_rate_khz == 48, "valid params")

cat("[validate_params] all valid rates...\n")
for (r in VALID_RATES) {
  p <- validate_params(1, 8, r)
  assert_true(p$sample_rate_khz == r, paste("rate", r, "accepted"))
}

cat("[validate_params] rejects invalid...\n")
assert_throws(function() validate_params(0, 8, 48), "n=0 throws")
assert_throws(function() validate_params(-1, 8, 48), "n=-1 throws")
assert_throws(function() validate_params(1, 0, 48), "bitrate=0 throws")
assert_throws(function() validate_params(1, 25, 48), "bitrate=25 throws")
assert_throws(function() validate_params(1, 8, 44), "rate=44 throws")

# ========== discretize ==========

cat("[discretize] correct output count...\n")
sig_const <- rep(10000.0, 100000)
out48 <- discretize(sig_const, 1, 8, 48)
assert_true(length(out48) == 48000, "48000 samples for 48KHz")

cat("[discretize] output count other rates...\n")
sig500k <- rep(5000.0, 500000)
out96 <- discretize(sig500k, 1, 8, 96)
assert_true(length(out96) == 96000, "96000 for 96KHz")
out256 <- discretize(sig500k, 1, 8, 256)
assert_true(length(out256) == 256000, "256000 for 256KHz")

cat("[discretize] constant signal preserves value...\n")
exact_level <- FREQ_MIN + 100 * step8
sig_exact <- rep(exact_level, 100000)
out_exact <- discretize(sig_exact, 1, 8, 48)
assert_true(all(abs(out_exact - exact_level) < 0.001), "constant preserved")

cat("[discretize] output values in range...\n")
sig_rand <- generate_signal(100000)
out_rand <- discretize(sig_rand, 1, 8, 48)
assert_true(all(out_rand >= FREQ_MIN - 0.001) && all(out_rand <= FREQ_MAX + 0.001),
            "all output in range")

cat("[discretize] output values on levels...\n")
check <- out_rand[1:1000]
levels_f <- (check - FREQ_MIN) / step8
assert_true(all(abs(levels_f - round(levels_f)) < 1e-6), "all on valid levels")

cat("[discretize] averaging...\n")
sig_pairs <- rep(c(100.0, 200.0), 48000)
out_avg <- discretize(sig_pairs, 1, 16, 48)
expected_avg <- quantize(150.0, step16)
assert_true(all(abs(out_avg[1:100] - expected_avg) < 0.01), "bins average correctly")

cat("[discretize] multi-second...\n")
sig_2s <- rep(10000.0, 200000)
out_2s <- discretize(sig_2s, 2, 8, 48)
assert_true(length(out_2s) == 96000, "2s * 48000 = 96000")

cat("[discretize] 1-bit quantization...\n")
sig_1b <- generate_signal(100000)
out_1b <- discretize(sig_1b, 1, 1, 48)
assert_true(all(abs(out_1b - FREQ_MIN) < 0.01 | abs(out_1b - FREQ_MAX) < 0.01),
            "1-bit only 20 or 20000")

cat("[discretize] higher bitrate less error...\n")
sig_precise <- rep(12345.6789, 100000)
out8 <- discretize(sig_precise, 1, 8, 48)
out16 <- discretize(sig_precise, 1, 16, 48)
assert_true(abs(out16[1] - 12345.6789) <= abs(out8[1] - 12345.6789),
            "16-bit error <= 8-bit error")

# ========== Results ==========

cat(sprintf("\n=== Results ===\nPassed: %d\nFailed: %d\n", tests_passed, tests_failed))
if (tests_failed > 0) quit(status = 1)
