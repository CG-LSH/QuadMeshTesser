@echo off
REM GPU 投影依赖（RTX 3090 / CUDA 12.x）
REM 与 numba 共用 numpy 1.26，避免 cupy 14 强制 numpy>=2
python -m pip install "numpy>=1.24,<2" "numba>=0.58" "cupy-cuda12x>=13.3,<14"
python -c "from quadmeshtesser.appr_parallel import available_backends, gpu_device_name; print('backends:', available_backends()); print('GPU:', gpu_device_name())"
pause
