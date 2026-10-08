## e1_thresholds.R — sensitivity of Procedure EQF(t) to the threshold scale and T_cons, per method.
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))), "common.R"))
out <- args_out(); e1 <- file.path(out, "E1")
D <- read.csv(file.path(e1, "results.csv"), stringsAsFactors = FALSE)
cols <- grep("^stop_s", names(D), value = TRUE)
Tmax <- max(c(D$stop_t, unlist(D[, cols])))
rows <- do.call(rbind, lapply(cols, function(cn) {
  s <- as.numeric(sub("stop_s([0-9.]+)_T.*", "\\1", cn)); Tc <- as.integer(sub(".*_T([0-9]+)$", "\\1", cn))
  do.call(rbind, lapply(split(D, D$method), function(S) {
    st <- S[[cn]]; fin <- st < Tmax
    data.frame(method = S$method[1], eps_scale = s, T_cons = Tc, stopped = mean(fin),
               median_stop = if (any(fin)) median(st[fin]) else NA, dist_at_T = median(S$final_dist),
               recognised_at_stop = mean(S$class_ok_stop)) })) }))
ensure_dir(file.path(e1, "tables")); ensure_dir(file.path(e1, "figures"))
write.csv(rows, file.path(e1, "tables", "e1_thresholds.csv"), row.names = FALSE); print(rows)
png(file.path(e1, "figures", "e1_threshold_sensitivity.png"), width = 1200, height = 450, res = 130)
par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))
cols2 <- c(baseline = "grey40", erim = "steelblue", ppo = "darkorange"); pch <- c(`4` = 1, `8` = 16, `16` = 2)
plot(NA, xlim = range(rows$eps_scale), ylim = c(0, 1), log = "x", xlab = "threshold scale", ylab = "fraction of missions stopped", main = "Stopping")
for (m in unique(rows$method)) for (Tc in unique(rows$T_cons)) { r <- rows[rows$method == m & rows$T_cons == Tc, ]; lines(r$eps_scale, r$stopped, col = cols2[m], type = "b", pch = pch[as.character(Tc)]) }
legend("bottomright", legend = c(unique(rows$method), paste("T_cons", unique(rows$T_cons))), col = c(cols2[unique(rows$method)], rep("black", length(unique(rows$T_cons)))), pch = c(rep(15, length(unique(rows$method))), pch[as.character(unique(rows$T_cons))]), bty = "n", cex = 0.8)
plot(NA, xlim = range(rows$eps_scale), ylim = c(0, Tmax), log = "x", xlab = "threshold scale", ylab = "median stopping stage", main = "When")
for (m in unique(rows$method)) for (Tc in unique(rows$T_cons)) { r <- rows[rows$method == m & rows$T_cons == Tc, ]; lines(r$eps_scale, r$median_stop, col = cols2[m], type = "b", pch = pch[as.character(Tc)]) }
dev.off()
