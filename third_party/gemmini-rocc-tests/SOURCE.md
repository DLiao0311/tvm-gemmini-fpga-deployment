# Gemmini software API source

- Upstream: `https://github.com/ucb-bar/gemmini-rocc-tests`
- Source commit: `1a1a1c6bd60df6d7cae3d87aac96c8f406cae084`
- Selected by Gemmini commit: `25809f78323a729ef76fb68f3cedd8a24da2942b`
- Validated with Chipyard: 1.13.0

## Local modification

The defaults for `DIM`, `BANK_ROWS`, and `ACC_ROWS` in `include/gemmini_params.h` are wrapped
with `#ifndef` guards. This allows Step 3 to supply values matching the selected FPGA Gemmini
configuration through compiler definitions.

The upstream copyright notice, redistribution conditions, and disclaimer are retained in
`LICENSE`.
