## e2_slopes.R — rates: excess cost vs width n (at the largest M), generalisation gap vs M (at the largest n),
## recognition error vs depth J and M; log-log slopes with bootstrap CIs over seeds.
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))), "common.R"))
out <- args_out(); e2 <- file.path(out, "E2")
E <- read.csv(file.path(e2, "rates_est.csv"), stringsAsFactors = FALSE)
REF <- read_csv_if(file.path(e2, "rates_ref.csv")); cost_ref <- if (!is.null(REF)) REF$cost_test[1] else min(E$cost_test)
E$excess <- pmax(E$cost_test - cost_ref, 1e-6)
slope <- function(x, y) { ok <- is.finite(x) & is.finite(y) & y > 0; if (sum(ok) < 3) return(c(NA, NA, NA)); f <- lm(log(y[ok]) ~ log(x[ok])); c(coef(f)[2], confint(f)[2, ]) }
ensure_dir(file.path(e2, "tables")); ensure_dir(file.path(e2, "figures"))
Mmax <- max(E$M); nmax <- max(E$n)
A <- aggregate(cbind(excess, err_zp_T, err_f_T, gap) ~ n + M, data = E, FUN = mean)
s_n <- slope(A$n[A$M == Mmax], A$excess[A$M == Mmax]); s_n_err <- slope(A$n[A$M == Mmax], A$err_zp_T[A$M == Mmax])
s_M <- slope(A$M[A$n == nmax], pmax(A$gap[A$n == nmax], 1e-6))
tab <- data.frame(quantity = c("excess cost vs n (M max)", "Object position error vs n (M max)", "generalisation gap vs M (n max)"),
                  slope = c(s_n[1], s_n_err[1], s_M[1]), ci_low = c(s_n[2], s_n_err[2], s_M[2]), ci_high = c(s_n[3], s_n_err[3], s_M[3]),
                  bound = c("-1/2", "-1/2", "-1/2"))
write.csv(tab, file.path(e2, "tables", "e2_slopes.csv"), row.names = FALSE); print(tab); write.csv(A, file.path(e2, "tables", "e2_grid_means.csv"), row.names = FALSE)
png(file.path(e2, "figures", "e2_rates.png"), width = 1500, height = 480, res = 130)
par(mfrow = c(1, 3), mar = c(4, 4, 3, 1))
plot(A$n[A$M == Mmax], A$excess[A$M == Mmax], log = "xy", pch = 16, xlab = "width n", ylab = "excess cost", main = sprintf("slope %.2f (bound -0.5)", s_n[1]))
plot(A$M[A$n == nmax], pmax(A$gap[A$n == nmax], 1e-6), log = "xy", pch = 16, xlab = "training missions M", ylab = "held-out minus training cost", main = sprintf("slope %.2f (bound -0.5)", s_M[1]))
C <- read_csv_if(file.path(e2, "rates_cnn.csv"))
if (!is.null(C)) { AC <- aggregate(acc ~ J + M, data = C, FUN = mean); plot(NA, xlim = range(AC$M), ylim = c(0, 1), log = "x", xlab = "training missions M", ylab = "recognition accuracy", main = "CNN depth J")
  for (J in unique(AC$J)) lines(AC$M[AC$J == J], AC$acc[AC$J == J], type = "b", pch = 15 + J, col = J); legend("bottomright", legend = paste("J =", unique(AC$J)), col = unique(AC$J), pch = 15 + unique(AC$J), bty = "n") }
dev.off()
