# GPU setup (training machine)

The development machine has **no NVIDIA GPU**, so everything below was prepared there and must be confirmed on the GPU machine with the smoke test.

What runs where:
- **Feature engineering** (pandas, numba) is CPU-only and takes minutes.
- **Model training** is what the GPU speeds up.
- **Serving** (`src/serving/`) runs on CPU.

## 1. Check the hardware and driver
```
nvidia-smi
```
Note the GPU name and the **CUDA Version** in the top-right corner. That is the newest CUDA runtime the driver supports.

## 2. Create the environment
```
python -m venv venv
venv\Scripts\activate                     # Linux: source venv/bin/activate
pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu128   # pick cuXXX <= driver CUDA
pip install -r requirements-gpu.txt
```
- **torch from plain PyPI on Windows is the CPU build.** That is the most common reason `torch.cuda.is_available()` returns `False`.
- **XGBoost and CatBoost:** the PyPI wheels already contain CUDA. CatBoost needs a driver new enough for its bundled CUDA runtime. Otherwise it fails with `CUDA error 35: CUDA driver version is insufficient`, and the fix is to update the NVIDIA driver.
- **LightGBM:** the PyPI wheel has the **OpenCL** GPU learner (`device_type="gpu"`), and this was verified on the development machine. NVIDIA's OpenCL comes with the driver.
  - The `cuda` learner needs a source build. The code tries `gpu`, then `cuda`, then CPU, and records which backend it used (`model.device_used_`).

## 3. Smoke test (zero errors required)
```
python scripts/gpu_smoke_test.py
```
It trains tiny models on synthetic data through the project's own training code. It captures native (C++) logs, so even XGBoost's silent "No visible GPU is found, setting device to CPU" counts as a failure.

Expected result: `RESULT: PASS`.

### If LightGBM picks the wrong GPU
OpenCL can list an integrated GPU before the NVIDIA card. Set these values in `configs/config.yaml` → `training_v2.lgbm` to the NVIDIA platform and device:
```yaml
gpu_platform_id: 0     # try 0/1 until the smoke test log shows the NVIDIA device
gpu_device_id: 0
```

### If LightGBM has no usable GPU backend
Run `python scripts/gpu_smoke_test.py --allow-lgbm-cpu`. LightGBM then trains on CPU, which is still fast for this data size, while XGBoost and CatBoost use the GPU.

## 4. Train on GPU
Add `--device gpu` to `run_training.py`, `run_ablation.py`, `run_tuning.py`, `run_heldout_pattern_test.py` and `run_sequence_experiment.py`. Or set `training_v2.device: gpu` in the config.

GPU-specific behaviour, all handled in `src/training/gbdt.py`:

| Library | GPU setting | Note |
|---|---|---|
| XGBoost | `device="cuda"`, `tree_method="hist"` | `gpu_hist` is deprecated. Prediction runs on CPU to avoid the device-mismatch copy and its warning. |
| CatBoost | `task_type="GPU"`, `devices="0"` | `PRAUC`/`AUC` are not implemented on GPU and would silently run on CPU every iteration, so on GPU early stopping uses `Logloss`. `thread_count` is removed. |
| LightGBM | `device_type="gpu"`, `max_bin=63` | OpenCL. Falls back to `cuda`, then CPU, and logs which one it used. |
| PyTorch GRU (Phase 9) | `cuda` if available | The model and every batch are moved to the device. |

The legacy VAE, BiLSTM and GNN (`src/models/`) are no longer part of the pipeline, so they were not GPU-hardened.
