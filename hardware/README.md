# Genesys 2 FPGA hardware flow

This project was validated on a Rocket Core + Gemmini SoC running Linux on a Digilent
Genesys 2 board. The hardware is generated with Chipyard 1.13.0 and implemented with
Vivado.

## Board-support dependency

Chipyard 1.13.0 does not provide the Genesys 2 target used by this project. The target is
supplied by the following public repository:

- Source: <https://github.com/stanley-666/chipyard_fpga_genesys2>
- Target Chipyard version: 1.13.0
- Licenses: BSD-3-Clause and Apache-2.0 (`LICENSE.SiFive`)
- Functionality stated by the upstream project: UART, JTAG, microSD, and the Genesys 2
  FPGA shell/harness

The validated hardware sources are published in the following forks and branches:

| Component | Repository / branch | Commit |
|---|---|---|
| Chipyard | `DLiao0311/chipyard`, `genesys2-gemmini-1.13.0` | `6f3015a5` |
| Gemmini | `DLiao0311/gemmini`, `genesys2-config` | `c8bcc68e` |
| FPGA shells | `DLiao0311/rocket-chip-fpga-shells`, `genesys2-support` | `83f9e9fe` |
| Original Genesys 2 port | `stanley-666/chipyard_fpga_genesys2` | `8a3aff1a` |

The Chipyard fork is based on Chipyard 1.13.0 commit
`69eba860a352343e4ac6b6df0f3638a79a86ec78`. Its submodule pointers and `.gitmodules`
URLs select the Gemmini and FPGA-shell commits listed above.

The board-support source is integrated into the attributed Chipyard and FPGA-shell forks; it
is not duplicated in this deployment repository. The upstream `LICENSE` and
`LICENSE.SiFive` notices are retained. Use the pinned commits above rather than the moving
branch tips for reproduction.

## Project-specific hardware configuration

The upstream repository provides the Genesys 2 board integration. This project adds the
Rocket + Gemmini configurations used by the deployment experiments, including Gemmini
array size, scratchpad capacity, accumulator capacity, and FPGA clock selection.

The validated local configuration variants are:

```text
Gemmini 4x4   / scratchpad 512 KiB / accumulator 256 KiB
Gemmini 8x8   / scratchpad 512 KiB / accumulator 256 KiB
Gemmini 16x16 / scratchpad 512 KiB / accumulator 256 KiB
Gemmini 32x32 / scratchpad 512 KiB / accumulator 256 KiB
```

The corresponding Scala configurations are committed in the Chipyard and Gemmini forks.
The Genesys 2 harness and shell remain attributed to the upstream board port; the four
resource-constrained Gemmini configurations are project-specific additions.

## Build sequence

```text
DLiao0311/chipyard @ 6f3015a5
        + DLiao0311/gemmini @ c8bcc68e
        + DLiao0311/rocket-chip-fpga-shells @ 83f9e9fe
        -> RTL generation
        -> Vivado project (.xpr)
        -> synthesis and preliminary utilization estimate
        -> implementation and post-route timing/utilization
        -> bitstream
        -> Linux boot on Rocket Core + Gemmini
```

Use Ubuntu 20.04 and the Vivado version from the validated FPGA machine when reproducing
the hardware build. The software compilation stages may run in a different compatible
Linux environment; they do not require Vivado.

## Files to preserve from a validated build

Record the following metadata with every published hardware result or bitstream:

- Chipyard commit
- Genesys 2 board-support commit
- Gemmini submodule commit
- project-specific Chipyard configuration name and patch
- Vivado version
- FPGA clock frequency
- synthesis/implementation utilization report
- post-route timing result
- bitstream checksum

Vivado projects, generated RTL, and checkpoints are intentionally not tracked. Selected
validated bitstreams are kept under `hardware/bit_stream/` for the current project archive.
Because these binaries increase clone size and Git cannot efficiently store later revisions,
future bitstream releases should preferably use GitHub Release assets. Curated reports may be
committed after removing machine-specific paths.

## Contribution boundary

The Genesys 2 FPGA shell and harness are credited to the upstream board-support project.
The work in this repository is the Gemmini configuration and the reproducible path from
model compilation through RISC-V/Linux execution on that hardware target.
