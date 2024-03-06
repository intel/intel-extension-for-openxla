# source /home/$USER/yangshe1/env.sh

export ZE_AFFINITY_MASK=0.0
# export ZE_AFFINITY_MASK=1
# export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1

# export ZE_ENABLE_TRACING_LAYER=1
# export UseCyclesPerSecondTimer=1
# export ENABLE_TF_PROFILER=1

# export ITEX_AUTO_MIXED_PRECISION=1
# export ITEX_AUTO_MIXED_PRECISION_DATA_TYPE="BFLOAT16"
# export ITEX_ONEDNN_LAYOUT_OPT=0
# export ITEX_REMAPPER=0
# export TF_XLA_FLAGS="--tf_xla_auto_jit=2"
# export TF_CPP_MAX_VLOG_LEVEL=1

rm -rf dump/*
# export XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_hlo_as_html --xla_dump_to=./dump --xla_dump_hlo_module_re=run_model"
# export XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_hlo_as_html --xla_dump_to=./dump --xla_dump_hlo_pass_re=attention"
# export XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_hlo_as_html --xla_dump_to=./dump --xla_dump_hlo_pass_re=attention --xla_dump_hlo_module_re=run_model"
# export XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=./dump --xla_disable_hlo_passes=dot-expand-dims"
# export XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_hlo_as_dot --xla_dump_to=./dump --xla_dump_hlo_module_re=run_model"
export XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=./dump"

# export PJRT_NAMES_AND_LIBRARY_PATHS="xpu:/home/$USER/yangshe1/libitex_xla_extension.so"
# export LD_LIBRARY_PATH="/home/$USER/miniconda3/envs/yang-jax/lib/python3.9/site-packages/jaxlib:$LD_LIBRARY_PATH"

# export ITEX_FP32_MATH_MODE=tf32

#export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1

export TF_LLVM_OPT=1
export XETLA_GEMM=1
export VECTORIZE=1

export MHA=1

# numactl -N 0 -m 0 \
# /home/$USER/yangshe1/pti-gpu/tools/ze_tracer/build/ze_tracer --chrome-device-timeline \
/home/$USER/yangshe1/pti-gpu/tools/ze_tracer/build/ze_tracer -t \
python jax_stable.py 2>&1 | tee log
# python jax_gptj.py --greedy --dtype "float16" 2>&1 | tee log
# python jax_gptj.py --dtype "float16" --num-iter 2 --num-warmup 1 2>&1 | tee log2
# python jax_gptj.py --input-tokens 1024 --max-new-tokens 128 --dtype "float16" --num-iter 2 --num-warmup 1 2>&1 | tee log3
# python jax_stable.py 2>&1 | tee log
# python jax_gptj_1024.py 2>&1 | tee log


# /home/$USER/yangshe1/pti-gpu/tools/ze_tracer/build/ze_tracer -t \
# python jax_gptj_1024.py 2>&1 | tee log_1024_4_gpu
#
# /home/$USER/yangshe1/pti-gpu/tools/ze_tracer/build/ze_tracer -t \
# python jax_gptj_1024_1.py 2>&1 | tee log_1024_1_gpu