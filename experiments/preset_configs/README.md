# Preset Configs

Pre-made configuration presets that serve as starting points for deploying RayFronts on different hardware and under different constraints.

> **Note:** These are suggestions only. Only the configurations in the paper appendix have been accurately measured and tested. We encourage you to play around with configs to achieve the best accuracy/compute/memory tradeoff for your specific use case.

For more guidance on tuning configs, see the [FAQ](https://github.com/RayFronts/RayFronts/wiki/FAQ).

## Available Presets

| Preset | Target | Description |
|--------|--------|-------------|
| `realtime_orinagx.yaml` | NVIDIA Jetson Orin AGX | Aggressive compute/memory reduction for embedded deployment. Uses PCA compression, low resolution, and limited point budgets. |
| `realtime_RTX4090.yaml` | Desktop GPU (RTX 4090) | Higher fidelity config for powerful desktop GPUs. Full resolution, no feature compression. |
| `low_mem.yaml` | Memory-constrained systems | Minimizes memory footprint with smallest resolution, aggressive PCA compression, and low point/ray budgets. |

## Usage

Use the `--config-dir` and `--config-name` flags to point the mapping server at a preset:

```bash
python3 -m rayfronts.mapping_server --config-dir experiments/preset_configs --config-name realtime_orinagx
```

You can override any parameter via the command line on top of the preset:

```bash
python3 -m rayfronts.mapping_server --config-dir experiments/preset_configs --config-name realtime_orinagx mapping.vox_size=0.5 dataset.frame_skip=5
```

## PCA Feature Compression

The `realtime_orinagx` and `low_mem` presets require a fitted PCA basis. Fit one on your target distribution using:

```bash
python scripts/fit_feat_compressor.py <args>
```

Set `feat_compressor.path` to the resulting file path.

## Customizing

These presets are intended as starting points. Key knobs to adjust:

- **Accuracy vs. speed:** `dataset.rgb_resolution`, `encoder.model_version` (base vs. large), `feat_compressor.out_dim`
- **Memory:** `mapping.max_pts_per_frame`, `mapping.max_rays_per_frame`, `mapping.angle_bin_size`, `feat_compressor.out_dim`
- **Latency:** `mapping.ray_tracing`, `mapping.vox_accum_period`, `dataset.frame_skip`, `vis: null`

See [configs](../../rayfronts/configs) for all available options.
