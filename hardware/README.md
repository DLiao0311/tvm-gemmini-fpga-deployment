# FPGA hardware flow

The original hardware workflow uses Chipyard to generate the Rocket Core + Gemmini design and
a Vivado project for the Digilent Genesys 2 board. Vivado is then used for synthesis,
implementation, resource/timing analysis, and bitstream generation.

```text
Chipyard configuration
        -> RTL generation
        -> Vivado project (.xpr)
        -> synthesis and preliminary utilization
        -> implementation and post-route timing/utilization
        -> bitstream
        -> Linux boot on Rocket Core + Gemmini
```

Generated Vivado projects, checkpoints, and bitstreams are intentionally not tracked. This
directory should eventually contain only the Chipyard configuration or patch, version metadata,
reproduction commands, and curated utilization/timing reports. A redistributable reference
bitstream may be attached to a GitHub Release instead of committed to Git.

Hardware configuration sources have not yet been imported because they are outside the original
`people_count` directory and must be identified from the Chipyard workspace.
