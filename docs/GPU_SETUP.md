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

## 3b. Running the whole pipeline: `run_all.py`
```
python run_all.py                       # full run (resumes automatically); --quick for a fast pass
python run_all.py --with-catboost       # CatBoost is off by default (blend weight 0, 2.8 h tuning in run 1)
python run_all.py --label-median-days 60 --tag-suffix _d60    # same pipeline under another chargeback-delay assumption
```
- **Python 3.11-3.13 only.** The bootstrap looks for such an interpreter (`py -3.13`, `python3.13` ...) to create `venv/`, and refuses 3.14 (`--allow-unsupported-python` overrides). On 3.14 the native GPU libraries crash at interpreter shutdown (0xC0000005). Delete an old 3.14 `venv/` first.
- Every script is launched through `src/utils/run_guard.py`: after `main()` returns it writes a success sentinel and ends the process with `os._exit`. A step with a sentinel counts as done even if the exit code is non-zero (warning in the log). "Out of memory" is only reported when the log shows a memory error.
- `results/run_all.lock` stops a second `run_all.py` from starting while one is active (stale locks of dead processes are taken over; `--force-lock` overrides).
- `run_training.py --final` reuses the validation run's cross-validation (rounds + out-of-fold predictions) when features, parameters, folds and label policy are unchanged; `--no-reuse-cv` forces a redo.

## 4. Train on GPU
Add `--device gpu` to `run_training.py`, `run_ablation.py`, `run_tuning.py`, `run_heldout_pattern_test.py` and `run_sequence_experiment.py`. Or set `training_v2.device: gpu` in the config.

GPU-specific behaviour, all handled in `src/training/gbdt.py`:

| Library | GPU setting | Note |
|---|---|---|
| XGBoost | `device="cuda"`, `tree_method="hist"` | `gpu_hist` is deprecated. Prediction runs on CPU to avoid the device-mismatch copy and its warning. |
| CatBoost | `task_type="GPU"`, `devices="0"` | `PRAUC`/`AUC` are CPU-only: they would run on the CPU every iteration, so on GPU early stopping uses `training_v2.catboost.gpu_eval_metric` (default `Logloss`; a proper scoring rule that correlates with, but is not, AP). `thread_count` is removed. CatBoost is off by default. |
| LightGBM | `device_type="gpu"`, `max_bin=255` (`training_v2.lgbm.gpu_max_bin`; 63 cost accuracy) | OpenCL. Falls back to `cuda`, then CPU, and logs which one it used. |
| PyTorch GRU (Phase 9) | `cuda` if available | The model and every batch are moved to the device. |

The legacy VAE, BiLSTM and GNN (`src/models/`) are no longer part of the pipeline, so they were not GPU-hardened.
