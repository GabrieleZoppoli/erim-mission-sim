## e4_curves.R — PPO learning curves and the comparison with E1 on the same missions.
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))), "common.R"))
out <- args_out(); e4 <- file.path(out, "E4")
L <- read.csv(file.path(e4, "learning_curve.csv"), stringsAsFactors = FALSE)
ensure_dir(file.path(e4, "figures")); ensure_dir(file.path(e4, "tables"))
png(file.path(e4, "figures", "e4_learning_curve.png"), width = 900, height = 450, res = 130); par(mar = c(4, 4, 3, 1))
plot(NA, xlim = range(L$env_steps), ylim = range(L$reward_mean), xlab = "environment steps", ylab = "mean reward per step", main = "PPO from pixels")
for (s in unique(L$seed)) lines(L$env_steps[L$seed == s], L$reward_mean[L$seed == s], col = s + 1, lwd = 2)
dev.off()
D4 <- read.csv(file.path(e4, "results.csv"), stringsAsFactors = FALSE); D1 <- read_csv_if(file.path(out, "E1", "results.csv"))
D <- if (!is.null(D1)) rbind(D1[, intersect(names(D1), names(D4))], D4[, intersect(names(D1), names(D4))]) else D4
tab <- do.call(rbind, lapply(split(D, D$method), function(S) data.frame(method = S$method[1], missions = nrow(S),
  recognised_T = sprintf("%.0f%%", 100 * mean(S$class_ok_T)), final_dist = fmt_ci(boot_ci(S$final_dist)), min_dist = fmt_ci(boot_ci(S$min_dist)),
  cost_per_stage = fmt_ci(boot_ci(S$cost_per_stage), 2), total_cost = fmt_ci(boot_ci(S$total_cost), 0))))
write.csv(tab, file.path(e4, "tables", "e4_vs_e1.csv"), row.names = FALSE); write_md_table(tab, file.path(e4, "tables", "e4_vs_e1.md")); print(tab)
