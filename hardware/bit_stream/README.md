# Genesys 2 bitstreams

These bitstreams are archived outputs from the Genesys 2 hardware design-space experiments.
The configuration encoded in each filename includes the system-bus width, FPGA clock,
Gemmini array organization, scratchpad capacity, accumulator capacity, and dataflow.

All files target the Digilent Genesys 2 board and must not be programmed onto a different
board. A bitstream filename alone is not a complete reproducibility record. The matching
Chipyard, Genesys 2 board-support, Gemmini, Vivado, configuration, timing, and utilization
metadata still need to be documented.

Bitstreams are generated artifacts and are not covered by the repository's Apache-2.0
statement unless explicitly stated otherwise. Use them only with the matching hardware and
at your own risk.
