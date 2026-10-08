## common.R — helpers for the analysis of erimsim outputs. Base R only.
## usage from the repo root:  Rscript R/e1_compare.R <out_dir>   (out_dir = the --out directory given to the runner)
args_out <- function() { a <- commandArgs(trailingOnly = TRUE); if (length(a) < 1) stop("usage: Rscript R/<script>.R <out_dir>"); a[1] }
read_csv_if <- function(path) if (file.exists(path)) read.csv(path, stringsAsFactors = FALSE) else NULL
boot_ci <- function(x, stat = median, B = 2000, seed = 1) {
  x <- x[is.finite(x)]; if (length(x) < 2) return(c(NA, NA, NA))
  set.seed(seed); s <- replicate(B, stat(sample(x, length(x), replace = TRUE)))
  c(stat(x), quantile(s, 0.025, names = FALSE), quantile(s, 0.975, names = FALSE))
}
fmt_ci <- function(ci, d = 3) sprintf(paste0("%.", d, "f [%.", d, "f, %.", d, "f]"), ci[1], ci[2], ci[3])
paired_boot <- function(x, y, B = 2000, seed = 1) {  # x - y, same missions
  ok <- is.finite(x) & is.finite(y); d <- x[ok] - y[ok]; set.seed(seed)
  s <- replicate(B, mean(sample(d, length(d), replace = TRUE))); c(mean(d), quantile(s, c(0.025, 0.975), names = FALSE))
}
method_label <- function(m) c(baseline = "EKF + LQ (CE)", erim = "ERIM (neural FSP)", ppo = "PPO from pixels")[m]
ensure_dir <- function(d) { dir.create(d, showWarnings = FALSE, recursive = TRUE); d }
write_md_table <- function(df, path) {
  con <- file(path, "w"); on.exit(close(con))
  writeLines(paste("|", paste(names(df), collapse = " | "), "|"), con)
  writeLines(paste("|", paste(rep("---", ncol(df)), collapse = " | "), "|"), con)
  for (i in seq_len(nrow(df))) writeLines(paste("|", paste(as.character(unlist(df[i, ])), collapse = " | "), "|"), con)
}
