#!/usr/bin/env bash
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

onednn_gpu_path="bazel-bin/external/onednn_gpu/include/oneapi/dnnl/dnnl_version.h"
onednn_cpu_path="bazel-bin/external/onednn_cpu/include/oneapi/dnnl/dnnl_version.h"
onednn_gpu_v2_path="bazel-bin/external/onednn_gpu_v2/include/oneapi/dnnl/dnnl_version.h"
onednn_cpu_v2_path="bazel-bin/external/onednn_cpu_v2/include/oneapi/dnnl/dnnl_version.h"
xla_tmp_folder_name="xla.tmp"

set -e

function is_absolute {
  [[ "$1" = /* ]] || [[ "$1" =~ ^[a-zA-Z]:[/\\].* ]]
}

function real_path() {
  is_absolute "$1" && echo "$1" || echo "$PWD/${1#./}"
}

function get_git_desc() {
  git_version=`git rev-parse --short=8 HEAD`
  echo $git_version
}

function get_compiler_version() {
  compiler_path=`cat .itex_configure.bazelrc | grep -Eo 'DPCPP_TOOLKIT_PATH=.*$' | cut -d '=' -f 2 | cut -d '"' -f 2`
  version=`${compiler_path}/bin/icx --version | grep -Eo '\([a-zA-Z0-9.]{10,}\)' | grep -Eo '[a-zA-Z0-9.]{10,}'`
  echo "dpcpp-${version}"
}

function get_onednn_git_version() {
  onednn_path=$1
  if [ ! -f ${onednn_path} ]; then
    echo "none"
  else
    major_version=`cat ${onednn_path} | grep '#define DNNL_VERSION_MAJOR' | cut -d ' ' -f 3`
    minor_version=`cat ${onednn_path} | grep '#define DNNL_VERSION_MINOR' | cut -d ' ' -f 3`
    patch_version=`cat ${onednn_path} | grep '#define DNNL_VERSION_PATCH' | cut -d ' ' -f 3`
    commit=`cat ${onednn_path} | grep '#define DNNL_VERSION_HASH' | grep -Eo '[a-z0-9]{40}'`
    version="v${major_version}.${minor_version}.${patch_version}-`echo ${commit} | cut -c 1-8`"
    echo $version
  fi
}

function emit_version_info() {
  if [ ! -f $1 ]; then
    echo "$1 not exists!"
    exit -1
  fi
  echo "__git_desc__= '`get_git_desc`'" >> $1
  echo "VERSION = __version__" >> $1
  echo "GIT_VERSION = 'v' + __version__ + '-' + __git_desc__" >> $1
  echo "COMPILER_VERSION = '`get_compiler_version`'" >> $1
  if [ -f ${onednn_gpu_path} ]; then
    onednn_path=${onednn_gpu_path}
  elif [ -f ${onednn_cpu_path} ]; then
    onednn_path=${onednn_cpu_path}
  elif [ -f ${onednn_gpu_v2_path} ]; then
    onednn_path=${onednn_gpu_v2_path}
  elif [ -f ${onednn_cpu_v2_path} ]; then
    onednn_path=${onednn_cpu_v2_path}
  else
    echo "Error: no oneDNN version files"
    exit -1
  fi
  onednn_git_version=`get_onednn_git_version ${onednn_path}`
  if [ ${onednn_git_version} != "none" ]; then
    echo "ONEDNN_GIT_VERSION = '${onednn_git_version}'" >> $1
  fi
  echo "TF_COMPATIBLE_VERSION = '>= 2.8.0'" >> $1
}


PLATFORM="$(uname -s | tr 'A-Z' 'a-z')"

function prepare_src() {
  if [ $# -lt 1 ] ; then
    echo "No destination dir provided"
    exit 1
  fi

  TMPDIR="$1"
  mkdir -p "$TMPDIR"
  XLA_TMPDIR="$TMPDIR/$xla_tmp_folder_name"
  mkdir -p "$XLA_TMPDIR"

  echo $(date) : "=== Preparing sources in dir: ${TMPDIR}"
  echo $(date) : "=== Preparing sources in dir: ${XLA_TMPDIR}"

  if [ ! -d bazel-bin/xla ]; then
    echo "Could not find bazel-bin.  Did you run from the root of the build tree?"
    exit 1
  fi

  RUNFILES=bazel-bin/xla/tools/pip_package/build_pip_package.runfiles/intel_extension_for_openxla
  cp -R \
      bazel-bin/xla/tools/pip_package/build_pip_package.runfiles/intel_extension_for_openxla/xla \
      "${XLA_TMPDIR}"
  
  # xla
  cp xla/tools/pip_package/xla_setup.py ${XLA_TMPDIR}/setup.py
  mkdir -p ${XLA_TMPDIR}/jax_plugins/openxla_xpu
  if [ -d ${XLA_TMPDIR}/xla ] ; then
    ls -al ${XLA_TMPDIR}
    mv -f ${XLA_TMPDIR}/xla/* ${XLA_TMPDIR}/jax_plugins/openxla_xpu/
    cp -rf xla/python/*.py ${XLA_TMPDIR}/jax_plugins/openxla_xpu/
    # emit_version_info ${XLA_TMPDIR}/jax_plugins/python/version.py
    chmod +x ${XLA_TMPDIR}/jax_plugins/openxla_xpu/__init__.py
    rm -rf ${XLA_TMPDIR}/xla
  fi
}

function build_wheel() {
  if [ $# -lt 2 ] ; then
    echo "No src and dest dir provided"
    exit 1
  fi

  TMPDIR="$1"
  DEST="$2"
  PKG_NAME_FLAG="$3"
  WEEKLY_BUILD_FLAG="$4"

  # Before we leave the top-level directory, make sure we know how to
  # call python.
  if [[ -e tools/python_bin_path.sh ]]; then
    source tools/python_bin_path.sh
  fi

  pushd ${TMPDIR}/${xla_tmp_folder_name} > /dev/null
  rm -f MANIFEST
  echo $(date) : "=== Building Intel® Extension for OpenXLA* wheel"
  "${PYTHON_BIN_PATH:-python}" setup.py bdist_wheel ${PKG_NAME_FLAG} ${WEEKLY_BUILD_FLAG}>/dev/null
  mkdir -p ${DEST}
  cp dist/* ${DEST}
  popd > /dev/null
  echo $(date) : "=== Output wheel path: ${DEST}"
}

function usage() {
  echo "Usage:"
  echo "$0 [--src srcdir] [--dst dstdir] [options]"
  echo "$0 dstdir [options]"
  echo ""
  echo "    --src                 prepare sources in srcdir"
  echo "                              will use temporary dir if not specified"
  echo ""
  echo "    --dst                 build wheel in dstdir"
  echo "                              if dstdir is not set do not build, only prepare sources"
  echo ""
  exit 1
}

function main() {
  PKG_NAME_FLAG=""
  PROJECT_NAME=""
  NIGHTLY_BUILD=0
  SRCDIR=""
  DSTDIR=""
  CLEANSRC=1
  WEEKLY_BUILD_FLAG=""
  while true; do
    if [[ "$1" == "--help" ]]; then
      usage
      exit 1
    elif [[ "$1" == "--project_name" ]]; then
      shift
      if [[ -z "$1" ]]; then
        break
      fi
      PROJECT_NAME="$1"
    elif [[ "$1" == "--src" ]]; then
      shift
      SRCDIR="$(real_path $1)"
      CLEANSRC=0
    elif [[ "$1" == "--weekly" ]]; then
      WEEKLY_BUILD_FLAG="--weekly_build"
    elif [[ "$1" == "--dst" ]]; then
      shift
      DSTDIR="$(real_path $1)"
    else
      DSTDIR="$(real_path $1)"
    fi
    shift

    if [[ -z "$1" ]]; then
      break
    fi
  done

  if [[ -z "$DSTDIR" ]] && [[ -z "$SRCDIR" ]]; then
    echo "No destination dir provided"
    usage
    exit 1
  fi

  if [[ -z "$SRCDIR" ]]; then
    # make temp srcdir if none set
    SRCDIR="$(mktemp -d -t tmp.XXXXXXXXXX)"
  fi

  prepare_src "$SRCDIR"

  if [[ -z "$DSTDIR" ]]; then
      # only want to prepare sources
      exit
  fi

  if [[ -n ${PROJECT_NAME} ]]; then
    PKG_NAME_FLAG="--project_name ${PROJECT_NAME}"
  fi

  build_wheel "$SRCDIR" "$DSTDIR" "$PKG_NAME_FLAG" "$WEEKLY_BUILD_FLAG"

  if [[ $CLEANSRC -ne 0 ]]; then
    rm -rf "${TMPDIR}"
  fi
}

main "$@"
