"""Run a model over the frozen eval set and emit metrics.

Reports cutoff_rate, added_latency_ms (p50/p95/p99), and false_hold_rate,
per ENGINEERING.md section 3. Replays force-aligned gold transcripts as a simulated
incremental stream -- never calls a live STT (see ENGINEERING.md section 10).

Not implemented yet. Arrives in the baselines phase.
"""
