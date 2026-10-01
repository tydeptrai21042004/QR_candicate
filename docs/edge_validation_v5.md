# v5 edge-runtime validation

Development reference, 512x512 `girl.bmp`, 64x64 watermark:

- original v4 full equivalence reference: ~23.57 s in this local Python environment;
- optimized v5 full equivalence run: ~0.41--0.48 s cold/single-run depending on cache state;
- warm-session benchmark: certified embed median ~198 ms (~5 fps), extraction median ~13.3 ms (~75 fps);
- full v4/v5 equivalence key `edge-full-equivalence-2026`: 0 differing image pixels, 0 differing packed period-code bytes, 0 differing packed certificate-mask bytes;
- all 48 tests pass;
- 20,000-sample hardware sanity: 0 rounded reconstruction mismatches and 0 QIM hard-decision mismatches.

These are development-CPU software measurements. They do not constitute an FPGA, ASIC, Raspberry Pi, Jetson, or other edge-board real-time benchmark.

For 512x512 at 30 fps, one base traversal plus four worst-case certificate traversals requires about 39.322 Mpixel/s when the two extreme 3x3 kernels execute in parallel. This is the engineering target for the streaming accelerator.
