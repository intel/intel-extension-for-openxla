/* Copyright (c) 2023 Intel Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/service/gpu/xetla/gemm/gemm.h"

#include "xla/service/gpu/matrix_descriptor.h"
#include "xla/service/gpu/xetla/gemm/dispatch_col_major.h"
#include "xla/service/gpu/xetla/gemm/dispatch_row_major.h"
#include "xla/service/gpu/xetla/gemm/gemm_common.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/gpu/gpu_types.h"
#include "xla/stream_executor/sycl/sycl_stream.h"

namespace se = ::stream_executor;

namespace gpu {
namespace xetla {

static std::unordered_map<std::string, std::tuple<int, int, int, int, int, int>>
    configMap = {{"1_4096_16384", std::make_tuple(128, 128, 16, 32, 64, 1)},
                 {"1_16384_4096", std::make_tuple(8, 512, 8, 16, 16, 1)},
                 {"1_4096_4096", std::make_tuple(128, 64, 16, 16, 64, 1)},
                 {"4_4096_16384", std::make_tuple(128, 64, 16, 16, 64, 1)},
                 {"4_16384_4096", std::make_tuple(8, 512, 8, 16, 16, 1)},
                 {"4_4096_4096", std::make_tuple(128, 64, 16, 16, 64, 1)},
                 {"4096_16384_4096", std::make_tuple(256, 256, 32, 64, 32, 1)},
                 {"4096_4096_4096", std::make_tuple(256, 256, 32, 64, 32, 1)},
                 {"4096_4096_16384", std::make_tuple(256, 256, 32, 64, 32, 1)},
                 {"32_4096_16384", std::make_tuple(32, 64, 8, 16, 16, 2)},
                 {"32_16384_4096", std::make_tuple(128, 512, 64, 32, 16, 1)},
                 {"32_4096_4096", std::make_tuple(128, 64, 16, 16, 64, 1)},
                 {"33_4096_16384", std::make_tuple(128, 64, 16, 16, 64, 1)},
                 {"33_16384_4096", std::make_tuple(128, 512, 64, 32, 16, 1)},
                 {"33_4096_4096", std::make_tuple(128, 64, 16, 16, 64, 1)},
                 {"64_4096_16384", std::make_tuple(128, 64, 16, 16, 64, 1)},
                 {"64_16384_4096", std::make_tuple(128, 256, 64, 16, 16, 1)},
                 {"64_4096_4096", std::make_tuple(128, 64, 16, 16, 64, 1)},
                 {"65_4096_16384", std::make_tuple(128, 64, 16, 16, 64, 1)},
                 {"65_16384_4096", std::make_tuple(128, 256, 64, 16, 16, 1)},
                 {"65_4096_4096", std::make_tuple(128, 64, 16, 16, 64, 1)},
                 {"128_4096_16384", std::make_tuple(128, 128, 32, 32, 32, 2)},
                 {"128_16384_4096", std::make_tuple(256, 256, 64, 32, 16, 1)},
                 {"128_4096_4096", std::make_tuple(128, 64, 16, 16, 64, 1)},
                 {"130_4096_16384", std::make_tuple(128, 128, 32, 32, 32, 2)},
                 {"130_16384_4096", std::make_tuple(256, 256, 64, 32, 16, 1)},
                 {"130_4096_4096", std::make_tuple(128, 128, 32, 32, 32, 2)},
                 {"256_4096_16384", std::make_tuple(128, 128, 32, 32, 32, 2)},
                 {"256_16384_4096", std::make_tuple(256, 256, 64, 32, 16, 1)},
                 {"256_4096_4096", std::make_tuple(128, 128, 32, 32, 32, 2)},
                 {"512_4096_16384", std::make_tuple(128, 256, 32, 32, 16, 1)},
                 {"512_16384_4096", std::make_tuple(256, 256, 64, 32, 16, 1)},
                 {"512_4096_4096", std::make_tuple(128, 128, 32, 32, 32, 2)},
                 {"513_4096_16384", std::make_tuple(256, 256, 64, 32, 16, 1)},
                 {"513_16384_4096", std::make_tuple(256, 256, 64, 32, 16, 1)},
                 {"513_4096_4096", std::make_tuple(128, 512, 64, 32, 16, 1)},
                 {"1024_4096_16384", std::make_tuple(256, 256, 64, 32, 16, 1)},
                 {"1024_16384_4096", std::make_tuple(256, 256, 64, 32, 16, 1)},
                 {"1024_4096_4096", std::make_tuple(256, 256, 64, 32, 16, 1)},
                 {"1028_4096_16384", std::make_tuple(256, 256, 64, 32, 16, 1)},
                 {"1028_16384_4096", std::make_tuple(256, 256, 64, 32, 16, 1)},
                 {"1028_4096_4096", std::make_tuple(256, 256, 64, 32, 16, 1)},
                 {"2016_4096_16384", std::make_tuple(256, 256, 64, 32, 16, 1)},
                 {"2016_16384_4096", std::make_tuple(256, 256, 64, 32, 16, 1)},
                 {"2016_4096_4096", std::make_tuple(256, 256, 64, 32, 16, 1)},
                 {"1_50400_4096", std::make_tuple(128, 512, 64, 32, 16, 1)},
                 {"1_50272_4096", std::make_tuple(128, 512, 64, 32, 16, 1)},
                 {"4_50400_4096", std::make_tuple(128, 512, 64, 32, 16, 1)},
                 {"4_50272_4096", std::make_tuple(128, 512, 64, 32, 16, 1)},
                 {"1_250880_4096", std::make_tuple(32, 64, 8, 16, 16, 2)},
                 {"4_250880_4096", std::make_tuple(32, 64, 8, 16, 16, 2)},
                 {"1_11008_4096", std::make_tuple(16, 256, 8, 16, 16, 1)},
                 {"4_11008_4096", std::make_tuple(16, 256, 8, 16, 16, 1)},
                 {"32_11008_4096", std::make_tuple(128, 256, 32, 32, 16, 1)},
                 {"64_11008_4096", std::make_tuple(64, 256, 64, 16, 16, 2)},
                 {"128_11008_4096", std::make_tuple(128, 256, 32, 32, 16, 1)},
                 {"256_11008_4096", std::make_tuple(256, 256, 32, 64, 32, 1)},
                 {"512_11008_4096", std::make_tuple(256, 256, 64, 32, 16, 1)},
                 {"1024_11008_4096", std::make_tuple(256, 256, 64, 32, 16, 1)},
                 {"2016_11008_4096", std::make_tuple(256, 256, 64, 32, 16, 1)},
                 {"1_32000_4096", std::make_tuple(256, 256, 32, 64, 16, 1)},
                 {"4_32000_4096", std::make_tuple(256, 256, 32, 64, 16, 1)},
                 {"1_13824_5120", std::make_tuple(256, 256, 64, 32, 16, 1)},
                 {"1_5120_5120", std::make_tuple(8, 128, 8, 16, 16, 2)},
                 {"4_13824_5120", std::make_tuple(128, 256, 64, 16, 16, 1)},
                 {"4_5120_5120", std::make_tuple(8, 128, 8, 16, 16, 2)},
                 {"32_13824_5120", std::make_tuple(128, 256, 64, 16, 16, 1)},
                 {"32_5120_5120", std::make_tuple(32, 128, 32, 16, 16, 4)},
                 {"64_13824_5120", std::make_tuple(128, 256, 32, 32, 16, 1)},
                 {"64_5120_5120", std::make_tuple(128, 128, 16, 32, 64, 1)},
                 {"128_13824_5120", std::make_tuple(128, 256, 32, 32, 16, 1)},
                 {"128_5120_5120", std::make_tuple(128, 128, 16, 32, 64, 1)},
                 {"256_13824_5120", std::make_tuple(256, 256, 32, 64, 32, 1)},
                 {"256_5120_5120", std::make_tuple(128, 256, 32, 32, 16, 1)},
                 {"512_13824_5120", std::make_tuple(256, 256, 32, 64, 32, 1)},
                 {"512_5120_5120", std::make_tuple(256, 256, 64, 32, 16, 1)},
                 {"1024_13824_5120", std::make_tuple(256, 256, 64, 32, 16, 1)},
                 {"1024_5120_5120", std::make_tuple(256, 256, 64, 32, 16, 1)},
                 {"2016_13824_5120", std::make_tuple(256, 256, 64, 32, 16, 1)},
                 {"2016_5120_5120", std::make_tuple(256, 256, 64, 32, 16, 1)},
                 {"1_32000_5120", std::make_tuple(256, 256, 32, 64, 16, 1)},
                 {"4_32000_5120", std::make_tuple(256, 256, 32, 64, 16, 1)},
                 {"1_7168_14336", std::make_tuple(8, 256, 8, 16, 16, 2)},
                 {"1_1792_14336", std::make_tuple(16, 64, 16, 16, 16, 8)},
                 {"4_7168_14336", std::make_tuple(8, 256, 8, 16, 16, 2)},
                 {"4_1792_14336", std::make_tuple(16, 64, 16, 16, 16, 8)},
                 {"32_7168_14336", std::make_tuple(16, 256, 16, 16, 16, 2)},
                 {"32_1792_14336", std::make_tuple(16, 64, 16, 16, 16, 8)},
                 {"33_7168_14336", std::make_tuple(32, 256, 32, 16, 16, 2)},
                 {"33_1792_14336", std::make_tuple(32, 64, 32, 16, 16, 8)},
                 {"64_7168_14336", std::make_tuple(128, 128, 16, 32, 64, 1)},
                 {"64_1792_14336", std::make_tuple(32, 64, 32, 16, 16, 8)},
                 {"65_7168_14336", std::make_tuple(128, 128, 16, 32, 64, 1)},
                 {"65_1792_14336", std::make_tuple(32, 64, 32, 16, 16, 8)},
                 {"1_14336_7168", std::make_tuple(128, 512, 64, 32, 16, 1)},
                 {"1_14336_1792", std::make_tuple(128, 256, 32, 32, 16, 1)},
                 {"4_14336_7168", std::make_tuple(128, 256, 64, 16, 16, 1)},
                 {"4_14336_1792", std::make_tuple(16, 256, 8, 16, 16, 1)},
                 {"32_14336_7168", std::make_tuple(256, 256, 64, 32, 16, 1)},
                 {"32_14336_1792", std::make_tuple(128, 256, 32, 32, 16, 1)},
                 {"33_14336_7168", std::make_tuple(128, 256, 64, 16, 16, 1)},
                 {"33_14336_1792", std::make_tuple(128, 256, 32, 32, 16, 1)},
                 {"64_14336_7168", std::make_tuple(128, 256, 32, 32, 16, 1)},
                 {"64_14336_1792", std::make_tuple(256, 256, 32, 64, 16, 1)},
                 {"65_14336_7168", std::make_tuple(128, 256, 32, 32, 16, 1)},
                 {"65_14336_1792", std::make_tuple(256, 256, 32, 64, 16, 1)},
                 {"1_250880_1792", std::make_tuple(32, 64, 8, 16, 16, 2)},
                 {"1_2048_8192", std::make_tuple(8, 64, 8, 16, 32, 8)},
                 {"1_3584_7168", std::make_tuple(32, 64, 8, 16, 16, 2)},
                 {"1_3584_8192", std::make_tuple(32, 64, 8, 16, 16, 2)},
                 {"1_7168_3584", std::make_tuple(128, 128, 16, 32, 64, 1)},
                 {"1_7168_8192", std::make_tuple(8, 256, 8, 16, 16, 2)},
                 {"1_8192_1024", std::make_tuple(128, 64, 16, 16, 64, 1)},
                 {"1_8192_2048", std::make_tuple(128, 64, 16, 16, 64, 1)},
                 {"1_8192_3584", std::make_tuple(8, 256, 8, 16, 16, 2)},
                 {"1_8192_7168", std::make_tuple(8, 256, 8, 16, 16, 2)},
                 {"1_256_8192", std::make_tuple(16, 64, 16, 16, 16, 8)},
                 {"1_32000_1024", std::make_tuple(256, 256, 32, 64, 16, 1)},
                 {"1_32000_2048", std::make_tuple(256, 256, 32, 64, 16, 1)},
                 {"4_2048_8192", std::make_tuple(8, 64, 8, 16, 32, 8)},
                 {"4_3584_7168", std::make_tuple(32, 64, 8, 16, 16, 2)},
                 {"4_3584_8192", std::make_tuple(32, 64, 8, 16, 16, 2)},
                 {"4_7168_3584", std::make_tuple(128, 128, 16, 32, 64, 1)},
                 {"4_7168_8192", std::make_tuple(8, 256, 8, 16, 16, 2)},
                 {"4_8192_1024", std::make_tuple(128, 64, 16, 16, 64, 1)},
                 {"4_8192_2048", std::make_tuple(128, 64, 16, 16, 64, 1)},
                 {"4_8192_3584", std::make_tuple(8, 256, 8, 16, 16, 2)},
                 {"4_8192_7168", std::make_tuple(8, 256, 8, 16, 16, 2)},
                 {"4_256_8192", std::make_tuple(16, 64, 16, 16, 16, 8)},
                 {"4_32000_1024", std::make_tuple(256, 256, 32, 64, 16, 1)},
                 {"4_32000_2048", std::make_tuple(256, 256, 32, 64, 16, 1)},
                 {"1024_2048_8192", std::make_tuple(128, 256, 32, 32, 16, 1)},
                 {"1024_7168_8192", std::make_tuple(256, 256, 64, 32, 16, 1)},
                 {"1024_8192_1024", std::make_tuple(256, 256, 64, 32, 16, 1)},
                 {"1024_8192_2048", std::make_tuple(256, 256, 64, 32, 16, 1)},
                 {"1024_8192_3584", std::make_tuple(256, 256, 32, 64, 32, 1)},
                 {"1024_8192_7168", std::make_tuple(256, 256, 32, 64, 32, 1)},
                 {"1024_256_8192", std::make_tuple(32, 128, 32, 16, 16, 4)},
                 {"1_2048_4096", std::make_tuple(8, 64, 8, 16, 32, 8)},
                 {"1_4096_2048", std::make_tuple(128, 64, 16, 16, 64, 1)},
                 {"1_4096_8192", std::make_tuple(128, 64, 16, 16, 64, 1)},
                 {"1_8192_4096", std::make_tuple(8, 256, 8, 16, 16, 2)},
                 {"4_2048_4096", std::make_tuple(16, 64, 16, 16, 16, 8)},
                 {"4_4096_2048", std::make_tuple(128, 64, 16, 16, 64, 1)},
                 {"4_4096_8192", std::make_tuple(128, 64, 16, 16, 64, 1)},
                 {"4_8192_4096", std::make_tuple(128, 256, 32, 32, 16, 1)},
                 {"32_2048_4096", std::make_tuple(16, 64, 16, 16, 16, 8)},
                 {"32_4096_2048", std::make_tuple(128, 64, 16, 16, 64, 1)},
                 {"32_4096_8192", std::make_tuple(128, 64, 16, 16, 64, 1)},
                 {"32_8192_4096", std::make_tuple(128, 64, 16, 16, 64, 1)},
                 {"1024_2048_4096", std::make_tuple(128, 128, 32, 32, 32, 2)},
                 {"1024_4096_2048", std::make_tuple(256, 256, 64, 32, 16, 1)},
                 {"1024_4096_8192", std::make_tuple(256, 256, 64, 32, 16, 1)},
                 {"1024_8192_4096", std::make_tuple(256, 256, 64, 32, 16, 1)},
                 {"1_2560_5120", std::make_tuple(32, 64, 32, 16, 16, 8)},
                 {"1_2560_8192", std::make_tuple(32, 64, 32, 16, 16, 8)},
                 {"1_5120_6912", std::make_tuple(128, 128, 16, 32, 64, 1)},
                 {"1_5120_2560", std::make_tuple(8, 128, 8, 16, 16, 2)},
                 {"1_6912_5120", std::make_tuple(128, 128, 16, 32, 64, 1)},
                 {"4_2560_5120", std::make_tuple(32, 64, 32, 16, 16, 8)},
                 {"4_2560_8192", std::make_tuple(32, 64, 32, 16, 16, 8)},
                 {"4_5120_6912", std::make_tuple(128, 128, 16, 32, 64, 1)},
                 {"4_5120_2560", std::make_tuple(8, 128, 8, 16, 16, 2)},
                 {"4_6912_5120", std::make_tuple(128, 128, 16, 32, 64, 1)},
                 {"32_2560_5120", std::make_tuple(32, 64, 32, 16, 16, 8)},
                 {"32_5120_6912", std::make_tuple(32, 128, 32, 16, 16, 4)},
                 {"32_5120_2560", std::make_tuple(32, 128, 32, 16, 16, 4)},
                 {"32_6912_5120", std::make_tuple(128, 128, 16, 32, 64, 1)},
                 {"33_2560_5120", std::make_tuple(128, 64, 16, 16, 64, 1)},
                 {"33_5120_6912", std::make_tuple(128, 128, 16, 32, 64, 1)},
                 {"33_5120_2560", std::make_tuple(128, 128, 16, 32, 64, 1)},
                 {"33_6912_5120", std::make_tuple(128, 128, 16, 32, 64, 1)},
                 {"1024_2560_5120", std::make_tuple(256, 256, 64, 32, 16, 1)},
                 {"1024_5120_6912", std::make_tuple(256, 256, 32, 64, 32, 1)},
                 {"1024_5120_2560", std::make_tuple(256, 256, 64, 32, 16, 1)},
                 {"1024_6912_5120", std::make_tuple(256, 256, 64, 32, 16, 1)},
                 {"1_32000_8192", std::make_tuple(256, 256, 32, 64, 16, 1)},
                 {"4_32000_8192", std::make_tuple(256, 256, 32, 64, 16, 1)},
                 {"32_7168_8192", std::make_tuple(128, 128, 16, 32, 64, 1)},
                 {"32_8192_7168", std::make_tuple(128, 256, 32, 32, 16, 1)},
                 {"32_2048_8192", std::make_tuple(16, 64, 16, 16, 16, 8)},
                 {"32_8192_2048", std::make_tuple(128, 64, 16, 16, 64, 1)},
                 {"1_5504_4096", std::make_tuple(64, 128, 64, 16, 16, 4)},
                 {"1_4096_5504", std::make_tuple(128, 128, 32, 32, 32, 2)},
                 {"1_2048_4096", std::make_tuple(16, 64, 16, 16, 16, 8)},
                 {"1_4096_2048", std::make_tuple(128, 64, 16, 16, 64, 1)},
                 {"4_5504_4096", std::make_tuple(8, 128, 8, 16, 16, 2)},
                 {"4_4096_5504", std::make_tuple(128, 64, 16, 16, 64, 1)},
                 {"4_2048_4096", std::make_tuple(16, 64, 16, 16, 16, 8)},
                 {"4_4096_2048", std::make_tuple(128, 64, 16, 16, 64, 1)},
                 {"32_5504_4096", std::make_tuple(32, 128, 32, 16, 16, 4)},
                 {"32_4096_5504", std::make_tuple(128, 64, 16, 16, 64, 1)},
                 {"32_2048_4096", std::make_tuple(16, 64, 16, 16, 16, 8)},
                 {"32_4096_2048", std::make_tuple(128, 64, 16, 16, 64, 1)},
                 {"1024_5504_4096", std::make_tuple(256, 256, 64, 32, 16, 1)},
                 {"1024_4096_5504", std::make_tuple(256, 256, 64, 32, 16, 1)},
                 {"1024_2048_4096", std::make_tuple(128, 128, 32, 32, 32, 2)},
                 {"1024_4096_2048", std::make_tuple(256, 256, 64, 32, 16, 1)},
                 {"1_50272_7168", std::make_tuple(128, 512, 64, 32, 16, 1)},
                 {"4_50272_7168", std::make_tuple(128, 512, 64, 32, 16, 1)},
                 {"32_3584_7168", std::make_tuple(32, 64, 8, 16, 16, 2)},
                 {"32_7168_3584", std::make_tuple(128, 128, 16, 32, 64, 1)},
                 {"1024_14336_7168", std::make_tuple(256, 256, 32, 64, 32, 1)},
                 {"1024_7168_14336", std::make_tuple(256, 256, 32, 64, 32, 1)},
                 {"1024_3584_7168", std::make_tuple(256, 256, 32, 64, 32, 1)},
                 {"1024_3584_8192", std::make_tuple(256, 256, 32, 64, 32, 1)},
                 {"1024_7168_3584", std::make_tuple(256, 256, 32, 64, 32, 1)},
                 {"1_5120_13696", std::make_tuple(32, 128, 8, 16, 32, 1)},
                 {"1_13696_5120", std::make_tuple(256, 256, 32, 64, 16, 1)},
                 {"4_5120_13696", std::make_tuple(8, 128, 8, 16, 16, 2)},
                 {"4_13696_5120", std::make_tuple(256, 256, 32, 64, 16, 1)},
                 {"16_5120_13696", std::make_tuple(64, 128, 64, 16, 16, 4)},
                 {"16_13696_5120", std::make_tuple(128, 256, 64, 16, 16, 1)},
                 {"32_5120_13696", std::make_tuple(32, 128, 32, 16, 16, 4)},
                 {"32_13696_5120", std::make_tuple(128, 256, 64, 16, 16, 1)},
                 {"1024_5120_13696", std::make_tuple(256, 256, 32, 64, 32, 1)},
                 {"1024_13696_5120", std::make_tuple(256, 256, 32, 64, 32, 1)},
                 {"1_125696_5120", std::make_tuple(256, 256, 32, 64, 16, 1)},
                 {"1_5120_125696", std::make_tuple(32, 128, 8, 16, 32, 1)},
                 {"4_125696_5120", std::make_tuple(256, 256, 32, 64, 16, 1)},
                 {"4_5120_125696", std::make_tuple(32, 128, 8, 16, 32, 1)},
                 {"32_4608_4096", std::make_tuple(32, 128, 32, 16, 16, 4)},
                 {"32_4096_4608", std::make_tuple(128, 64, 16, 16, 64, 1)},
                 {"32_4096_13696", std::make_tuple(128, 64, 16, 16, 64, 1)},
                 {"32_13696_4096", std::make_tuple(128, 256, 64, 16, 16, 1)},
                 {"1024_4608_4096", std::make_tuple(256, 256, 32, 64, 32, 1)},
                 {"1024_4096_4608", std::make_tuple(256, 256, 32, 64, 32, 1)},
                 {"1024_4096_13696", std::make_tuple(256, 256, 32, 64, 32, 1)},
                 {"1024_13696_4096", std::make_tuple(256, 256, 32, 64, 32, 1)},
                 {"1_65024_4096", std::make_tuple(256, 256, 32, 64, 16, 1)},
                 {"1_4096_65024", std::make_tuple(128, 128, 16, 32, 64, 1)},
                 {"1_4608_4096", std::make_tuple(8, 32, 8, 16, 16, 4)},
                 {"1_4096_4608", std::make_tuple(128, 64, 16, 16, 64, 1)},
                 {"1_13696_4096", std::make_tuple(256, 256, 32, 64, 16, 1)},
                 {"1_4096_13696", std::make_tuple(128, 64, 16, 16, 64, 1)},
                 {"4_65024_4096", std::make_tuple(256, 256, 32, 64, 16, 1)},
                 {"4_4096_65024", std::make_tuple(128, 128, 16, 32, 64, 1)},
                 {"4_4608_4096", std::make_tuple(32, 128, 32, 16, 16, 4)},
                 {"4_4096_4608", std::make_tuple(128, 64, 16, 16, 64, 1)},
                 {"4_13696_4096", std::make_tuple(128, 256, 64, 16, 16, 1)},
                 {"4_4096_13696", std::make_tuple(128, 64, 16, 16, 64, 1)},
                 // T5 model shape
                 {"1_5120_2048", std::make_tuple(128, 128, 16, 32, 64, 1)},
                 {"1_2048_2048", std::make_tuple(16, 64, 16, 16, 16, 8)},
                 {"1_2048_5120", std::make_tuple(16, 64, 16, 16, 16, 8)},
                 {"1_6144_2048", std::make_tuple(128, 128, 16, 32, 64, 1)},
                 {"1_2048_2048", std::make_tuple(32, 64, 32, 16, 16, 8)},
                 {"32_98304_2048", std::make_tuple(256, 256, 32, 64, 16, 1)},
                 {"64_1_32", std::make_tuple(8, 256, 8, 16, 16, 2)},
                 {"1_32_64", std::make_tuple(8, 256, 8, 16, 16, 2)},
                 {"1_32128_2048", std::make_tuple(128, 256, 64, 16, 16, 1)},
                 {"32_5120_2048", std::make_tuple(128, 128, 16, 32, 64, 1)},
                 {"32_24576_32", std::make_tuple(32, 512, 32, 16, 16, 1)},
                 {"32_6144_2048", std::make_tuple(128, 128, 16, 32, 64, 1)},
                 {"32_2048_5120", std::make_tuple(16, 64, 16, 16, 16, 8)},
                 {"32_2048_2048", std::make_tuple(16, 64, 16, 16, 16, 8)}};

std::tuple<int, int, int, int, int, int> selectXetlaGemmConfig(int m, int n,
                                                               int k) {
  std::string mnk =
      std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k);
  if (configMap.find(mnk) != configMap.end()) {
    return configMap[mnk];
  }
  // TODO: optimize auto-tuning algorithm
  if (n == 4096 && m <= 128) {
    return std::make_tuple(128, 64, 16, 16, 64, 1);
  } else if (m >= 64) {
    if (m <= 512 && n <= 5120) {
      return std::make_tuple(128, 128, 32, 32, 32, 2);
    } else {
      return std::make_tuple(256, 256, 32, 64, 16, 1);
    }
  }
  // default config
  return std::make_tuple(16, 64, 16, 16, 16, 8);
}

std::tuple<int, int, int, int, int, int> selectXetlaQKVGemmConfig(int m, int n,
                                                                  int k) {
  return std::make_tuple(256, 256, 32, 64, 16, 1);
}

template <typename ComputeType>
bool XetlaGemmKernel<ComputeType>::run(se::gpu::GpuStreamHandle handle) {
  DispatchParams params(a_, b_, c_, m_, n_, k_, alpha_, num_epilogues_,
                        epilogue_tensors_, epilogue_types_, epilogue_params_);
  if (is_b_row_major_) {
    return GemmRowMajorDispatcher<ComputeType>(&params, selected_policy_id_)
        .run(handle);
  }
  return GemmColMajorDispatcher<ComputeType>(&params, selected_policy_id_)
      .run(handle);
}

template class XetlaGemmKernel<sycl::half>;
template class XetlaGemmKernel<gpu::xetla::bf16>;

}  // namespace xetla
}  // namespace gpu
