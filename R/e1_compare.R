## e1_compare.R — main comparison: medians with bootstrap CIs per method, paired differences on identical missions,
## figures. Reads <out>/E1/results.csv (and E4/results.csv if present); writes <out>/E1/tables/*, <out>/E1/figures/*.
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))), "common.R"))
out <- args_out(); e1 <- file.path(out, "E1")
D <- read.csv(file.path(e1, "results.csv"), stringsAsFactors = FALSE)
D4 <- read_csv_if(file.path(out, "E4", "results.csv")); if (!is.null(D4)) D <- rbind(D[, intersect(names(D), names(D4))], D4[, intersect(names(D), names(D4))])
D$stop_t_fin <- ifelse(D$stopped == 1, D$stop_t, NA)
metrics <- c(final_dist = "final distance (m)", min_dist = "minimum distance (m)", cost_per_stage = "cost per stage",
             err_zp_T = "Object position error at T (m)", err_zv_T = "Object velocity error at T", err_f_T = "f error at T",
             err_g_T = "g error at T", err_xp_T = "Machine position error at T (m)", err_xa_T = "Machine attitude error at T (rad)",
             t_first_correct = "first stage with correct class", stop_t_fin = "stopping stage (stopped missions)",
             dist_stop = "distance at stop (m)", err_zp_stop = "Object position error at stop (m)")
tab <- do.call(rbind, lapply(split(D, D$method), function(S) {
  r <- data.frame(method = S$method[1], missions = nrow(S), seeds = length(unique(S$seed)),
                  stopped = sprintf("%.0f%%", 100 * mean(S$stopped)), recognised_T = sprintf("%.0f%%", 100 * mean(S$class_ok_T)),
                  recognised_stop = sprintf("%.0f%%", 100 * mean(S$class_ok_stop)))
  for (m in names(metrics)) r[[m]] <- fmt_ci(boot_ci(S[[m]]))
  r }))
rownames(tab) <- NULL
ensure_dir(file.path(e1, "tables")); ensure_dir(file.path(e1, "figures"))
write.csv(tab, file.path(e1, "tables", "e1_summary.csv"), row.names = FALSE); write_md_table(tab, file.path(e1, "tables", "e1_summary.md"))
print(tab[, c("method", "missions", "stopped", "recognised_T", "final_dist", "cost_per_stage", "err_zp_T", "err_f_T")])
## paired differences ERIM - baseline on the same missions (ERIM averaged over seeds first)
if (all(c("erim", "baseline") %in% D$method)) {
  E <- aggregate(D[D$method == "erim", c("mission", "total_cost", "final_dist", "err_zp_T", "err_f_T", "err_g_T", "class_ok_T", "stop_t")],
                 by = list(mission = D$mission[D$method == "erim"]), FUN = mean)[, -1]
  B <- D[D$method == "baseline", ]; B <- B[match(E$mission, B$mission), ]
  pd <- do.call(rbind, lapply(c("total_cost", "final_dist", "err_zp_T", "err_f_T", "err_g_T", "class_ok_T", "stop_t"), function(m) {
    ci <- paired_boot(E[[m]], B[[m]]); data.frame(metric = m, mean_diff_erim_minus_baseline = ci[1], ci_low = ci[2], ci_high = ci[3]) }))
  write.csv(pd, file.path(e1, "tables", "e1_paired.csv"), row.names = FALSE); print(pd)
}
## figures
png(file.path(e1, "figures", "e1_boxplots.png"), width = 1600, height = 500, res = 130)
par(mfrow = c(1, 4), mar = c(6, 4, 3, 1))
for (m in c("final_dist", "cost_per_stage", "err_zp_T", "err_f_T"))
  boxplot(D[[m]] ~ D$method, xlab = "", ylab = m, main = metrics[m], las = 2, col = c("grey85", "lightsteelblue", "wheat")[seq_along(unique(D$method))])
dev.off()
png(file.path(e1, "figures", "e1_stop_hist.png"), width = 1200, height = 450, res = 130)
ms <- unique(D$method); par(mfrow = c(1, length(ms)), mar = c(4, 4, 3, 1))
for (m in ms) { x <- D$stop_t_fin[D$method == m]; if (sum(is.finite(x)) > 1) hist(x, breaks = 20, col = "grey85", main = paste("Stopping stage,", m), xlab = "t_f from Procedure EQF(t)") else plot.new() }
dev.off()
tr <- read_csv_if(file.path(e1, "trajectories_sample.csv"))
if (!is.null(tr)) {
  png(file.path(e1, "figures", "e1_recognition_vs_time.png"), width = 900, height = 450, res = 130)
  par(mar = c(4, 4, 3, 1)); plot(NA, xlim = range(tr$t), ylim = c(0, 1), xlab = "stage t", ylab = "posterior of the true class", main = "Recognition along the mission (sample missions)")
  cols <- c(baseline = "grey40", erim = "steelblue", ppo = "darkorange")
  for (m in unique(tr$method)) { A <- aggregate(post_true ~ t, data = tr[tr$method == m, ], FUN = mean); lines(A$t, A$post_true, col = cols[m], lwd = 2) }
  legend("bottomright", legend = unique(tr$method), col = cols[unique(tr$method)], lwd = 2, bty = "n")
  dev.off()
}
cat("written", file.path(e1, "tables"), "and", file.path(e1, "figures"), "\n")
