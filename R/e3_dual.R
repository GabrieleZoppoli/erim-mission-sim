## e3_dual.R — dual effect: control cost against information collected for each lambda, recognition, distance;
## paired differences against lambda = 0 on identical missions; sample trajectories.
## Reads <out>/E3/results.csv (total cost including lambda * sum_t entropy) and, when present,
## <out>/E3/results_control_cost.csv (the same policies re-evaluated with lambda = 0: control cost and entropy_sum,
## written by python -m erimsim.experiments.e3_reeval). Writes <out>/E3/tables/* and <out>/E3/figures/*.
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))), "common.R"))
out <- args_out(); e3 <- file.path(out, "E3")
D <- read.csv(file.path(e3, "results.csv"), stringsAsFactors = FALSE)
D$t_first_correct[D$t_first_correct < 0] <- NA
C <- read_csv_if(file.path(e3, "results_control_cost.csv"))
if (!is.null(C)) {
  key <- function(X) paste(X$lambda, X$seed, X$mission)
  i <- match(key(D), key(C))
  D$control_cost <- C$total_cost[i]; D$entropy_sum <- C$entropy_sum[i]; D$entropy_T <- C$entropy_T[i]
  D$info_term <- D$lambda * D$entropy_sum
  ok <- is.finite(D$control_cost)
  rel <- abs(D$control_cost[ok] + D$info_term[ok] - D$total_cost[ok]) / pmax(1, abs(D$total_cost[ok]))
  cat(sprintf("identity total = control + lambda * entropy_sum: %d of %d rows matched, max relative deviation %.2e\n", sum(ok), nrow(D), max(rel)))
} else {
  cat("results_control_cost.csv not found: control cost and information term not separated (total cost only)\n")
  D$control_cost <- NA; D$entropy_sum <- NA; D$entropy_T <- NA; D$info_term <- NA
}
ensure_dir(file.path(e3, "tables")); ensure_dir(file.path(e3, "figures"))
summarise <- function(S, by) data.frame(by, missions = nrow(S), seeds = length(unique(S$seed)),
  recognised_T = sprintf("%.1f%%", 100 * mean(S$class_ok_T)), t_first_correct = fmt_ci(boot_ci(S$t_first_correct), 0),
  final_dist = fmt_ci(boot_ci(S$final_dist)), total_cost = fmt_ci(boot_ci(S$total_cost, stat = mean), 1),
  control_cost = fmt_ci(boot_ci(S$control_cost, stat = mean), 1), entropy_sum = fmt_ci(boot_ci(S$entropy_sum, stat = mean), 1),
  info_term = sprintf("%.1f", mean(S$info_term)), entropy_T = fmt_ci(boot_ci(S$entropy_T, stat = mean), 3),
  err_zp_T = fmt_ci(boot_ci(S$err_zp_T)), stringsAsFactors = FALSE)
tab <- do.call(rbind, lapply(split(D, D$lambda), function(S) summarise(S, data.frame(lambda = S$lambda[1]))))
tab_seed <- do.call(rbind, lapply(split(D, list(D$lambda, D$seed), drop = TRUE), function(S) summarise(S, data.frame(lambda = S$lambda[1], seed = S$seed[1]))))
rownames(tab) <- NULL; rownames(tab_seed) <- NULL
write.csv(tab, file.path(e3, "tables", "e3_summary.csv"), row.names = FALSE); write_md_table(tab, file.path(e3, "tables", "e3_summary.md"))
write.csv(tab_seed, file.path(e3, "tables", "e3_summary_by_seed.csv"), row.names = FALSE)
print(tab[, c("lambda", "missions", "recognised_T", "final_dist", "total_cost", "control_cost", "entropy_sum", "info_term")])
## paired differences against lambda = 0 on the same missions (seeds averaged within lambda first)
lams <- sort(unique(D$lambda))
if (0 %in% lams && length(lams) > 1) {
  agg <- function(l) { S <- D[D$lambda == l, c("mission", "total_cost", "control_cost", "entropy_sum", "entropy_T", "class_ok_T", "final_dist", "err_zp_T")]
    A <- aggregate(S[, -1], by = list(mission = S$mission), FUN = mean); A[order(A$mission), ] }
  A0 <- agg(0)
  pd <- do.call(rbind, lapply(lams[lams != 0], function(l) { Al <- agg(l); Al <- Al[match(A0$mission, Al$mission), ]
    do.call(rbind, lapply(c("total_cost", "control_cost", "entropy_sum", "entropy_T", "class_ok_T", "final_dist", "err_zp_T"), function(m) {
      ci <- paired_boot(Al[[m]], A0[[m]]); data.frame(lambda = l, metric = m, mean_diff_vs_lambda0 = ci[1], ci_low = ci[2], ci_high = ci[3]) })) }))
  write.csv(pd, file.path(e3, "tables", "e3_paired_vs_lambda0.csv"), row.names = FALSE); print(pd)
}
## figure: the trade-off, information collected (lower entropy sum) against control cost, per lambda and seed
if (!is.null(C)) {
  M <- aggregate(cbind(control_cost, entropy_sum) ~ lambda + seed, data = D, FUN = mean)
  Ml <- aggregate(cbind(control_cost, entropy_sum) ~ lambda, data = D, FUN = mean)
  png(file.path(e3, "figures", "e3_tradeoff.png"), width = 900, height = 600, res = 130); par(mar = c(4.5, 4.5, 3, 1))
  cols <- setNames(c("grey40", "steelblue", "darkorange", "firebrick", "darkgreen")[seq_along(lams)], lams)
  plot(M$entropy_sum, M$control_cost, pch = 1, col = cols[as.character(M$lambda)], xlab = "sum over stages of the posterior entropy (nats)",
       ylab = "control cost (squared distance + control effort)", main = "Dual effect: information collected against control cost")
  points(Ml$entropy_sum, Ml$control_cost, pch = 16, cex = 1.6, col = cols[as.character(Ml$lambda)])
  o <- order(Ml$lambda); lines(Ml$entropy_sum[o], Ml$control_cost[o], col = "grey60", lty = 2)
  legend("topright", legend = paste("lambda =", lams), col = cols, pch = 16, bty = "n")
  dev.off()
}
## sample trajectories: posterior of the true class and distance along the mission, by lambda
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
cat("written", file.path(e3, "tables"), "and", file.path(e3, "figures"), "\n")
