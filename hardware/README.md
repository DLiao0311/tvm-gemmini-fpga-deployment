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

The board-support source is not copied into this repository. Follow its installation
instructions and preserve its `LICENSE` and `LICENSE.SiFive` files. For a reproducible
build, record the exact commit used rather than relying permanently on its moving
`master` branch.

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

The corresponding Scala configuration files still need to be extracted from the
validated Chipyard workspace and added here as a small, reviewable project-specific
patch. They should not be presented as part of the upstream Genesys 2 port.

## Build sequence

```text
Chipyard 1.13.0
        + Genesys 2 board-support repository
        + project-specific Rocket/Gemmini Config
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
