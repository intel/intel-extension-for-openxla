#! /bin/bash
# A script for single-node pile pretraining
set -x

if [ ! -n "$1" ];then
  echo "Error: the format is ./quick_start.sh [model size] [dataset dir] [model dir] [input_length] [output_length] [device type] [is_profile] [profile out dir]!"
  exit 1
fi

pushd ./t5x

mkdir logs
mkdir output

export T5X_DIR=`pwd`
export T5X_WORKSPACE_DIR=${T5X_DIR}/workspace
LOG_DIR=${T5X_DIR}/logs
export PYTHONPATH=${T5X_DIR}
export TFDS_DATA_DIR=$2
MODEL_SIZE=$1
MODEL_DIR=$3
INPUT_LENGTH=$4
OUTPUT_LENGTH=$5
export DEVICE_TYPE="XPU"  # XPU or CUDA
if [ -n "$6" ];then
  DEVICE_TYPE=$6
fi
export IS_PROFILE=$7
export PROFILE_DIR=$8
if [ ! -n "$4" ];then
  INPUT_LENGTH=32
fi
if [ ! -n "$5" ];then
  OUTPUT_LENGTH=32
fi
if [ ! -n "$8" ];then
   mkdir -p ${PROFILE_DIR}
fi

export PTI_ENABLE_COLLECTION=0


sed -i 's/"inputs": .*, "targets": .*}/"inputs": '${INPUT_LENGTH}', "targets": '${OUTPUT_LENGTH}'}/g' ${T5X_DIR}/../xl_infer.gin
sed -i 's/"inputs": .*, "targets": .*}/"inputs": '${INPUT_LENGTH}', "targets": '${OUTPUT_LENGTH}'}/g' ${T5X_DIR}/../xxl_infer.gin


# Arguments
PREC="bfloat16"        # Precision (float32, float16, bfloat16)


NUM_GPUS=1      # Number of GPUs (1, 2, 4, 8)
BSIZE_PER_GPU=1 # Batch size per GPU (varies with model size)
T5_NAME=flan-t5-$MODEL_SIZE
GIN_FILE="${T5X_DIR}/../xl_infer.gin"
MODEL_PATH=${MODEL_DIR}/checkpoint_1138000

if [ ${MODEL_SIZE} == "xxl" ];then
  GIN_FILE="${T5X_DIR}/../xxl_infer.gin"
  MODEL_PATH=${MODEL_DIR}/checkpoint_1114000
fi

echo $MODEL_PATH

echo "Please make sure ${NUM_GPUS} is the number of visible CUDA devices you have"

# Setting XLA flags
export XLA_FLAGS="--xla_gpu_simplify_all_fp_conversions --xla_gpu_all_reduce_combine_threshold_bytes=136314880 ${XLA_FLAGS}"


PREFIX=""
if [ -n "${IS_PROFILE}" ];then
  echo " MODE: Profile"
  export XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_hlo_as_html ${XLA_FLAGS}"
  export XLA_FLAGS="--xla_dump_to=${PROFILE_DIR} ${XLA_FLAGS}"
  #export XLA_FLAGS="--xla_dump_hlo_pass_re=attention ${XLA_FLAGS}"
  if [ ${DEVICE_TYPE} == "XPU" ];then
    PREFIX='unitrace --conditional-collection --chrome-device-logging --demangle --output-dir-path '${PROFILE_DIR}
    echo  ${PREFIX}
    #exit
  fi
fi

# Global batch size
BSIZE=$(( NUM_GPUS * BSIZE_PER_GPU  ))

${PREFIX} \
python3 -u ${T5X_DIR}/t5x/infer.py \
  --gin_file=${GIN_FILE} \
  --gin.CHECKPOINT_PATH=\"${MODEL_PATH}\" \
  --gin.network.T5Config.dtype=\"${PREC}\" \
  --gin.utils.DatasetConfig.batch_size=${BSIZE} \
  --gin.INFER_OUTPUT_DIR=\"${T5X_DIR}/output\" \
  --gin.seqio.SentencePieceVocabulary.sentencepiece_model_file=\"${MODEL_DIR}/sentencepiece.model\" \
  |& tee ${LOG_DIR}/${T5_NAME}_gpu_${NUM_GPUS}_${PREC}_gbs_${BSIZE}.log

popd
