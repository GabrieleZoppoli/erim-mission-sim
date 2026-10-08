"""Experiments E1-E4. Each module exposes units(cfg) -> list of unit ids and run_unit(cfg, unit_id, out) that appends
its rows to CSV files under out/<exp>/ and returns a short dict for the log. Units are independent and resumable."""
