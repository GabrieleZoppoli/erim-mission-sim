## e4_curves.R — PPO from pixels: learning curves of every run (E4 and any E4_s*/E4 directory under <out>), the
## best-iterate markers, and the evaluation of final and best iterates per run next to the two chains of E1 on the
## same missions. The reward is minus the stage cost over 100, so the curves are drawn as cost per stage on a log axis.
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))), "common.R"))
out <- args_out(); e4 <- file.path(out, "E4")
runs <- c(main = e4); extra <- list.files(out, pattern = "^E4_s", full.names = TRUE); for (d in extra) if (file.exists(file.path(d, "E4", "learning_curve.csv"))) runs[basename(d)] <- file.path(d, "E4")
L <- do.call(rbind, lapply(names(runs), function(r) { x <- read.csv(file.path(runs[r], "learning_curve.csv"), stringsAsFactors = FALSE); x$run <- r; x$id <- paste0(r, " s", x$seed); x }))
Bst <- do.call(rbind, lapply(names(runs), function(r) { b <- read_csv_if(file.path(runs[r], "ppo_best.csv")); if (is.null(b)) NULL else { b$run <- r; b$id <- paste0(r, " s", b$seed); b } }))
L$cost_per_stage <- pmax(-100 * L$reward_mean, 1e-2)
ensure_dir(file.path(e4, "figures")); ensure_dir(file.path(e4, "tables"))
ids <- unique(L$id); cols <- setNames(c("firebrick", "steelblue", "darkorange", "darkgreen", "purple")[seq_along(ids)], ids)
png(file.path(e4, "figures", "e4_learning_curve.png"), width = 1400, height = 520, res = 130); par(mfrow = c(1, 2), mar = c(4, 4.5, 3, 1))
plot(NA, xlim = range(L$update), ylim = c(min(3, min(L$cost_per_stage)), max(L$cost_per_stage)), log = "y", xlab = "PPO update", ylab = "training cost per stage (log)", main = "PPO from pixels: whole run")
for (i in ids) lines(L$update[L$id == i], L$cost_per_stage[L$id == i], col = cols[i], lwd = 1.5)
if (!is.null(Bst)) for (k in seq_len(nrow(Bst))) { Ls <- L[L$id == Bst$id[k], ]; j <- which.min(abs(Ls$update - Bst$best_update[k])); points(Ls$update[j], Ls$cost_per_stage[j], pch = 21, bg = "white", col = cols[Bst$id[k]], cex = 1.4) }
abline(h = 5.09, lty = 3); text(0, 5.09, "ERIM policy at evaluation (5.09)", pos = 3, cex = 0.75, adj = 0)
legend("topleft", legend = c(ids, "best iterate"), col = c(cols[ids], "black"), lwd = c(rep(1.5, length(ids)), NA), pch = c(rep(NA, length(ids)), 21), bty = "n", cex = 0.85)
Z <- L[L$update <= 600, ]
plot(NA, xlim = c(0, 600), ylim = range(Z$cost_per_stage), log = "y", xlab = "PPO update", ylab = "training cost per stage (log)", main = "First 600 updates")
for (i in ids) lines(Z$update[Z$id == i], Z$cost_per_stage[Z$id == i], col = cols[i], lwd = 1.5)
if (!is.null(Bst)) for (k in seq_len(nrow(Bst))) if (Bst$best_update[k] <= 600) abline(v = Bst$best_update[k], col = cols[Bst$id[k]], lty = 3)
dev.off()
## evaluation table: per run and iterate, next to E1
D4 <- do.call(rbind, lapply(names(runs), function(r) { x <- read.csv(file.path(runs[r], "results.csv"), stringsAsFactors = FALSE); x$run <- r; x }))
D1 <- read_csv_if(file.path(out, "E1", "results.csv")); if (!is.null(D1)) { D1$run <- "E1"; D1 <- D1[D1$method %in% c("baseline", "erim"), ] }
cm <- intersect(names(D4), if (is.null(D1)) names(D4) else names(D1)); D <- if (!is.null(D1)) rbind(D1[, cm], D4[, cm]) else D4[, cm]
D$group <- ifelse(D$method %in% c("baseline", "erim"), D$method, paste0(D$method, " ", D$run, " s", D$seed))
tab <- do.call(rbind, lapply(split(D, D$group), function(S) data.frame(policy = S$group[1], missions = nrow(S),
  final_dist = fmt_ci(boot_ci(S$final_dist)), min_dist = fmt_ci(boot_ci(S$min_dist)), cost_per_stage = fmt_ci(boot_ci(S$cost_per_stage), 2), total_cost = fmt_ci(boot_ci(S$total_cost, stat = mean), 0))))
rownames(tab) <- NULL
if (!is.null(Bst)) tab$best_update <- sapply(tab$policy, function(g) { k <- which(paste0("ppo_best ", Bst$id) == g); if (length(k)) Bst$best_update[k] else NA })
write.csv(tab, file.path(e4, "tables", "e4_vs_e1.csv"), row.names = FALSE); write_md_table(tab, file.path(e4, "tables", "e4_vs_e1.md")); print(tab)
cat("note: PPO has no recogniser or estimator; only distance and cost columns are meaningful for it\n")
