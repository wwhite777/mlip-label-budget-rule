# Local release checks

Run on the pristine public distribution and restored archive with the pinned light CPU environment. Set native numerical threads to one and use at most ten compute workers; the documented checks use seven.

- Check both public protocols, scientific inventories and code identities with `tools/check_protocol.py`.
- Verify every restored scientific file and the complete predecessor chain with `tools/check_stages.py`.
- Recompute all reported readings and both judges with `tools/replay_readings.py`.
- Recompute the descriptive mechanism tables with `tools/replay_mechanism.py`.
- Check generated README tables with `reporting/readme_tables.py --check`.
- Run the statistical, draw, guard, source-mutation and predecessor-mutation tests in `tests/`.
- Run the original synthetic matrix, generated-input CPU-R matrix and judge fixtures using the README commands. Synthetic results never authorize scientific conclusions.
- Scan every file in the repository and archive trees using `tools/check_public.py --all-files`; run its planted controls with `--self-test`. Controls run in isolated temporary scratch. Scan numeric-array metadata as well as text, compressed text and image/document metadata.
- Check repository and archive `SHA256SUMS` and exact inventories. Do not add weights, local environments or execution outputs to the pristine trees.
- Materialize deterministic tar parts in a fresh directory, verify all part and chunk hashes, restore into a fresh directory and check every reconstructed file hash. The archive tools include corruption, truncation, path and destination controls.

These are construction and reproduction checks. Publication is a separate action. The exact-original-content limitation in `TRANSFORMATIONS.md` remains explicit.
