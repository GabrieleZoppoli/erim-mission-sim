## e2_slopes.R — numerical check of the rates in the error bound. Excess closed-loop cost and excess Object position
## error over the reference training (rates_ref.csv: n_ref, M_ref, J_ref) against the width n at the largest M and
## against the training missions M at the largest n; recognition error of the CNN against M at the largest depth and
## against J at the largest M. Log-log slopes with cluster-bootstrap CIs over the training seeds (each cell's seeds are
## resampled with replacement, cell means refitted). The bound's reference rates are -1/2 in n and in M; the fitted
## slopes are compared with them as an upper bound on the rate, not as a prediction of it.
## The held-out-minus-training-missions cost difference ("gap") is reported, not fitted: both sets are held out from
## the policy search and the difference is within mission-sampling noise.
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))), "common.R"))
out <- args_out(); e2 <- file.path(out, "E2")
E <- read.csv(file.path(e2, "rates_est.csv"), stringsAsFactors = FALSE)
REF <- read_csv_if(file.path(e2, "rates_ref.csv"))
if (is.null(REF)) { cat("rates_ref.csv missing: using the best grid cell as the reference (provisional)\n")
  best <- aggregate(cbind(cost_test, err_zp_T, err_f_T) ~ n + M, data = E, FUN = mean); best <- best[which.min(best$cost_test), ]
  REF <- data.frame(n = best$n, M = best$M, cost_test = best$cost_test, err_zp_T = best$err_zp_T, err_f_T = best$err_f_T, cnn_acc = NA, cnn_J = NA, cnn_M = NA) }
cost_ref <- REF$cost_test[1]; ezp_ref <- REF$err_zp_T[1]; ef_ref <- REF$err_f_T[1]
FLOOR_COST <- 0.5; FLOOR_ERR <- 0.002     # floors for a cell mean that falls below the single-seed reference (noise)
ensure_dir(file.path(e2, "tables")); ensure_dir(file.path(e2, "figures"))
Mmax <- max(E$M); nmax <- max(E$n)
cell_means <- function(D) aggregate(cbind(cost_test, cost_train, gap, err_zp_T, err_f_T) ~ n + M, data = D, FUN = mean)
slope_fit <- function(x, y) { ok <- is.finite(x) & is.finite(y) & y > 0; if (sum(ok) < 3) return(NA); unname(coef(lm(log(y[ok]) ~ log(x[ok])))[2]) }
excess <- function(A) data.frame(A, ex_cost = pmax(A$cost_test - cost_ref, FLOOR_COST), ex_zp = pmax(A$err_zp_T - ezp_ref, FLOOR_ERR), ex_f = pmax(A$err_f_T - ef_ref, FLOOR_ERR))
fits <- function(A) { A <- excess(A)
  c(cost_n = slope_fit(A$n[A$M == Mmax], A$ex_cost[A$M == Mmax]), cost_M = slope_fit(A$M[A$n == nmax], A$ex_cost[A$n == nmax]),
    zp_n = slope_fit(A$n[A$M == Mmax], A$ex_zp[A$M == Mmax]), zp_M = slope_fit(A$M[A$n == nmax], A$ex_zp[A$n == nmax]),
    f_n = slope_fit(A$n[A$M == Mmax], A$ex_f[A$M == Mmax]), f_M = slope_fit(A$M[A$n == nmax], A$ex_f[A$n == nmax])) }
A0 <- cell_means(E); s0 <- fits(A0)
set.seed(1); B <- 2000
boot <- replicate(B, { Db <- do.call(rbind, lapply(split(E, list(E$n, E$M), drop = TRUE), function(S) S[sample(nrow(S), nrow(S), replace = TRUE), ])); fits(cell_means(Db)) })
ci <- apply(boot, 1, quantile, c(0.025, 0.975), na.rm = TRUE)
A0x <- excess(A0); floored <- sum(A0x$cost_test - cost_ref < FLOOR_COST) + sum(A0x$err_zp_T - ezp_ref < FLOOR_ERR)
tab <- data.frame(quantity = c("excess cost vs n (M max)", "excess cost vs M (n max)", "excess Object position error vs n (M max)", "excess Object position error vs M (n max)",
                               "excess f error vs n (M max)", "excess f error vs M (n max)"),
                  slope = round(s0, 3), ci_low = round(ci[1, ], 3), ci_high = round(ci[2, ], 3), bound_rate = -0.5,
                  reference = sprintf("n=%d, M=%d: cost %.1f, err_zp %.3f, err_f %.3f", REF$n[1], REF$M[1], cost_ref, ezp_ref, ef_ref))
write.csv(tab, file.path(e2, "tables", "e2_slopes.csv"), row.names = FALSE); write_md_table(tab[, 1:5], file.path(e2, "tables", "e2_slopes.md")); print(tab[, 1:5])
cat(sprintf("cells floored at the reference: %d of %d\n", floored, 2 * nrow(A0x)))
write.csv(A0x, file.path(e2, "tables", "e2_grid_means.csv"), row.names = FALSE)
## held-out minus training-missions cost: reported, not fitted
G <- aggregate(gap ~ n, data = E[E$M == Mmax, ], FUN = function(g) c(mean = mean(g), sd = sd(g))); G <- data.frame(n = G$n, gap_mean = G$gap[, 1], gap_sd = G$gap[, 2])
write.csv(G, file.path(e2, "tables", "e2_gap.csv"), row.names = FALSE); cat("held-out minus training-missions cost at M max (mean, sd over seeds):\n"); print(G)
## CNN: recognition error against M at the largest J and against J at the largest M (single-image accuracy, mid-range and overall)
C <- read_csv_if(file.path(e2, "rates_cnn.csv")); cnn_tab <- NULL
if (!is.null(C)) {
  AC <- aggregate(cbind(acc, acc_near, acc_mid, acc_far) ~ J + M, data = C, FUN = mean); Jmax <- max(AC$J); McM <- max(AC$M)
  err_M <- slope_fit(AC$M[AC$J == Jmax], 1 - AC$acc[AC$J == Jmax]); err_mid_M <- slope_fit(AC$M[AC$J == Jmax], 1 - AC$acc_mid[AC$J == Jmax])
  err_J <- slope_fit(AC$J[AC$M == McM], 1 - AC$acc[AC$M == McM]); err_mid_J <- slope_fit(AC$J[AC$M == McM], 1 - AC$acc_mid[AC$M == McM])
  bootc <- replicate(B, { Cb <- do.call(rbind, lapply(split(C, list(C$J, C$M), drop = TRUE), function(S) S[sample(nrow(S), nrow(S), replace = TRUE), ]))
    ACb <- aggregate(cbind(acc, acc_mid) ~ J + M, data = Cb, FUN = mean)
    c(slope_fit(ACb$M[ACb$J == Jmax], 1 - ACb$acc[ACb$J == Jmax]), slope_fit(ACb$M[ACb$J == Jmax], 1 - ACb$acc_mid[ACb$J == Jmax]),
      slope_fit(ACb$J[ACb$M == McM], 1 - ACb$acc[ACb$M == McM]), slope_fit(ACb$J[ACb$M == McM], 1 - ACb$acc_mid[ACb$M == McM])) })
  cic <- apply(bootc, 1, quantile, c(0.025, 0.975), na.rm = TRUE)
  cnn_tab <- data.frame(quantity = c("CNN error (all ranges) vs M (J max)", "CNN error (mid range) vs M (J max)", "CNN error (all ranges) vs J (M max)", "CNN error (mid range) vs J (M max)"),
                        slope = round(c(err_M, err_mid_M, err_J, err_mid_J), 3), ci_low = round(cic[1, ], 3), ci_high = round(cic[2, ], 3))
  if (!is.na(REF$cnn_acc[1])) cnn_tab$reference <- sprintf("J=%s, M=%s: acc %.3f", REF$cnn_J[1], REF$cnn_M[1], REF$cnn_acc[1])
  write.csv(cnn_tab, file.path(e2, "tables", "e2_cnn_slopes.csv"), row.names = FALSE); write.csv(AC, file.path(e2, "tables", "e2_cnn_grid.csv"), row.names = FALSE); print(cnn_tab)
}
## figure: excess cost and excess error against n and M (log-log, reference lines of slope -1/2), CNN accuracy
png(file.path(e2, "figures", "e2_rates.png"), width = 1600, height = 1000, res = 130); par(mfrow = c(2, 3), mar = c(4, 4.2, 3, 1))
refline <- function(x, y, s = -0.5) { i <- which.min(x); lines(range(x), y[i] * (range(x) / x[i])^s, lty = 2, col = "grey50") }
pan <- function(x, y, xl, yl, main) { plot(x, y, log = "xy", pch = 16, type = "b", xlab = xl, ylab = yl, main = main); refline(x, y) }
pan(A0x$n[A0x$M == Mmax], A0x$ex_cost[A0x$M == Mmax], "width n", "excess cost over the reference", sprintf("cost vs n at M=%d: slope %.2f", Mmax, s0["cost_n"]))
pan(A0x$M[A0x$n == nmax], A0x$ex_cost[A0x$n == nmax], "training missions M", "excess cost over the reference", sprintf("cost vs M at n=%d: slope %.2f", nmax, s0["cost_M"]))
pan(A0x$n[A0x$M == Mmax], A0x$ex_zp[A0x$M == Mmax], "width n", "excess Object position error (m)", sprintf("err_zp vs n: slope %.2f", s0["zp_n"]))
pan(A0x$M[A0x$n == nmax], A0x$ex_zp[A0x$n == nmax], "training missions M", "excess Object position error (m)", sprintf("err_zp vs M: slope %.2f", s0["zp_M"]))
if (!is.null(C)) { plot(NA, xlim = range(AC$M), ylim = c(0, 1), log = "x", xlab = "training missions M", ylab = "single-image accuracy", main = "CNN: depth J (solid: all ranges, dotted: 4-8 m)")
  for (J in sort(unique(AC$J))) { lines(AC$M[AC$J == J], AC$acc[AC$J == J], type = "b", pch = 15 + J, col = J); lines(AC$M[AC$J == J], AC$acc_mid[AC$J == J], lty = 3, col = J) }
  if (!is.na(REF$cnn_acc[1])) points(REF$cnn_M[1], REF$cnn_acc[1], pch = 8, cex = 1.5)
  legend("topleft", legend = c(paste("J =", sort(unique(AC$J))), "reference"), col = c(sort(unique(AC$J)), 1), pch = c(15 + sort(unique(AC$J)), 8), bty = "n", cex = 0.85)
  plot(A0x$n[A0x$M == Mmax], A0x$ex_f[A0x$M == Mmax], log = "xy", pch = 16, type = "b", xlab = "width n", ylab = "excess f error", main = sprintf("err_f vs n: slope %.2f", s0["f_n"])); refline(A0x$n[A0x$M == Mmax], A0x$ex_f[A0x$M == Mmax]) }
dev.off()
cat("written", file.path(e2, "tables"), "and", file.path(e2, "figures"), "\n")
