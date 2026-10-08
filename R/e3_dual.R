## e3_dual.R — dual effect: recognition time, cost and distance against lambda; sample trajectories.
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))), "common.R"))
out <- args_out(); e3 <- file.path(out, "E3")
D <- read.csv(file.path(e3, "results.csv"), stringsAsFactors = FALSE)
D$t_first_correct[D$t_first_correct < 0] <- NA
tab <- do.call(rbind, lapply(split(D, D$lambda), function(S) data.frame(lambda = S$lambda[1], missions = nrow(S),
  recognised_T = sprintf("%.0f%%", 100 * mean(S$class_ok_T)), t_first_correct = fmt_ci(boot_ci(S$t_first_correct), 0),
  final_dist = fmt_ci(boot_ci(S$final_dist)), cost_per_stage = fmt_ci(boot_ci(S$cost_per_stage), 2), err_zp_T = fmt_ci(boot_ci(S$err_zp_T)))))
ensure_dir(file.path(e3, "tables")); ensure_dir(file.path(e3, "figures"))
write.csv(tab, file.path(e3, "tables", "e3_summary.csv"), row.names = FALSE); write_md_table(tab, file.path(e3, "tables", "e3_summary.md")); print(tab)
tr <- read_csv_if(file.path(e3, "trajectories_sample.csv"))
if (!is.null(tr)) {
  png(file.path(e3, "figures", "e3_dual.png"), width = 1200, height = 450, res = 130); par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))
  ms <- unique(tr$method); cols <- seq_along(ms); names(cols) <- ms
  plot(NA, xlim = range(tr$t), ylim = c(0, 1), xlab = "stage", ylab = "posterior of the true class", main = "Recognition")
  for (m in ms) { A <- aggregate(post_true ~ t, data = tr[tr$method == m, ], FUN = mean); lines(A$t, A$post_true, col = cols[m], lwd = 2) }
  legend("bottomright", legend = ms, col = cols, lwd = 2, bty = "n")
  plot(NA, xlim = range(tr$t), ylim = c(0, max(tr$dist)), xlab = "stage", ylab = "Machine-Object distance (m)", main = "Approach")
  for (m in ms) { A <- aggregate(dist ~ t, data = tr[tr$method == m, ], FUN = mean); lines(A$t, A$dist, col = cols[m], lwd = 2) }
  dev.off()
}
