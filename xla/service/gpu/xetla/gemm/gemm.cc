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
#include "xla/service/gpu/xetla/gemm/hgemm_impl.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/gpu/gpu_types.h"
#include "xla/stream_executor/sycl/sycl_stream.h"

namespace se = ::stream_executor;

namespace gpu {
namespace xetla {

std::unordered_map<std::string, std::tuple<int, int, int, int, int, int>>
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
template <int WG_M, int WG_N, int SG_M, int SG_N, int SG_K, int SLM_KS,
          bool B_ROW_MAJOR>
bool XetlaGemmKernel<ComputeType>::dispatch(se::gpu::GpuStreamHandle handle) {
  sycl::queue q = *handle;
  if (num_epilogues_ == 0) {
    hgemm_common<ComputeType, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, 1, 1, 3,
                 B_ROW_MAJOR>(
        q, reinterpret_cast<ComputeType*>(c_->data.opaque()),
        reinterpret_cast<ComputeType*>(a_->data.opaque()),
        reinterpret_cast<ComputeType*>(b_->data.opaque()), m_, n_, k_);
  } else if (num_epilogues_ == 1 && epilogue_types_[0] == RES_ADD) {
    if (alpha_ == 1.0f) {
      hgemm_res<ComputeType, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, 1, 1, 3,
                B_ROW_MAJOR>(
          q, reinterpret_cast<ComputeType*>(c_->data.opaque()),
          reinterpret_cast<ComputeType*>(a_->data.opaque()),
          reinterpret_cast<ComputeType*>(b_->data.opaque()),
          reinterpret_cast<ComputeType*>(epilogue_tensors_[0]), m_, n_, k_,
          epilogue_params_[0]);
    } else {
      hgemm_addmm<ComputeType, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, 1, 1, 3,
                  B_ROW_MAJOR>(
          q, reinterpret_cast<ComputeType*>(c_->data.opaque()),
          reinterpret_cast<ComputeType*>(epilogue_tensors_[0]),
          reinterpret_cast<ComputeType*>(a_->data.opaque()),
          reinterpret_cast<ComputeType*>(b_->data.opaque()), m_, n_, k_, alpha_,
          epilogue_params_[0]);
    }
  } else if (num_epilogues_ == 1 && epilogue_types_[0] == GELU) {
    CHECK(alpha_ == 1.0f);
    hgemm_gelu<ComputeType, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, 1, 1, 3,
               B_ROW_MAJOR>(
        q, reinterpret_cast<ComputeType*>(c_->data.opaque()),
        reinterpret_cast<ComputeType*>(a_->data.opaque()),
        reinterpret_cast<ComputeType*>(b_->data.opaque()), m_, n_, k_);
  } else if (num_epilogues_ == 1 && epilogue_types_[0] == BIAS) {
    CHECK(alpha_ == 1.0f);
    hgemm_bias<ComputeType, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, 1, 1, 3,
               B_ROW_MAJOR>(
        q, reinterpret_cast<ComputeType*>(c_->data.opaque()),
        reinterpret_cast<ComputeType*>(a_->data.opaque()),
        reinterpret_cast<ComputeType*>(b_->data.opaque()),
        reinterpret_cast<ComputeType*>(epilogue_tensors_[0]), m_, n_, k_,
        epilogue_params_[0]);
  } else if (num_epilogues_ == 2 && epilogue_types_[0] == BIAS &&
             epilogue_types_[1] == RES_ADD) {
    CHECK(alpha_ == 1.0f);
    hgemm_bias_res<ComputeType, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, 1, 1, 3,
                   B_ROW_MAJOR>(
        q, reinterpret_cast<ComputeType*>(c_->data.opaque()),
        reinterpret_cast<ComputeType*>(a_->data.opaque()),
        reinterpret_cast<ComputeType*>(b_->data.opaque()),
        reinterpret_cast<ComputeType*>(epilogue_tensors_[0]),
        reinterpret_cast<ComputeType*>(epilogue_tensors_[1]), m_, n_, k_,
        epilogue_params_[0], epilogue_params_[1]);
  } else if (num_epilogues_ == 2 && epilogue_types_[0] == BIAS &&
             epilogue_types_[1] == GELU) {
    CHECK(alpha_ == 1.0f);
    hgemm_bias_gelu<ComputeType, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, 1, 1, 3,
                    B_ROW_MAJOR>(
        q, reinterpret_cast<ComputeType*>(c_->data.opaque()),
        reinterpret_cast<ComputeType*>(a_->data.opaque()),
        reinterpret_cast<ComputeType*>(b_->data.opaque()),
        reinterpret_cast<ComputeType*>(epilogue_tensors_[0]), m_, n_, k_,
        epilogue_params_[0]);

  } else {
    LOG(ERROR) << "No mateched policy, will fallback to oneDNN kernel";
    return false;
  }
  return true;
}

template <typename ComputeType, int WG_M, int WG_N, int SG_M, int SG_N,
          int SG_K, int SLM_KS>
struct GemmPolicy {
  static bool match_or_call(int wg_m, int wg_n, int sg_m, int sg_n, int sg_k,
                            int slm_ks, bool b_row_major,
                            XetlaGemmKernel<ComputeType>* gemm_kernel,
                            se::gpu::GpuStreamHandle handle) {
    if (WG_M == wg_m && WG_N == wg_n && SG_M == sg_m && SG_N == sg_n &&
        SG_K == sg_k && SLM_KS == slm_ks) {
      if (b_row_major) {
        return gemm_kernel
            ->template dispatch<WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, true>(
                handle);
      }
      return gemm_kernel
          ->template dispatch<WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, false>(
              handle);
    }
    return false;
  }
};

template <typename ComputeType, typename MATCHER, typename... TArgs>
struct PolicyDispatcher {
  static bool call(int wg_m, int wg_n, int sg_m, int sg_n, int sg_k, int slm_ks,
                   bool b_row_major, XetlaGemmKernel<ComputeType>* gemm_kernel,
                   se::gpu::GpuStreamHandle handle) {
    if (MATCHER::match_or_call(wg_m, wg_n, sg_m, sg_n, sg_k, slm_ks,
                               b_row_major, gemm_kernel, handle)) {
      return true;
    }
    return PolicyDispatcher<ComputeType, TArgs...>::call(
        wg_m, wg_n, sg_m, sg_n, sg_k, slm_ks, b_row_major, gemm_kernel, handle);
  }
};

template <typename ComputeType, typename MATCHER>
struct PolicyDispatcher<ComputeType, MATCHER> {
  static bool call(int wg_m, int wg_n, int sg_m, int sg_n, int sg_k, int slm_ks,
                   bool b_row_major, XetlaGemmKernel<ComputeType>* gemm_kernel,
                   se::gpu::GpuStreamHandle handle) {
    if (MATCHER::match_or_call(wg_m, wg_n, sg_m, sg_n, sg_k, slm_ks,
                               b_row_major, gemm_kernel, handle)) {
      return true;
    }
    return false;
  }
};

template <typename ComputeType>
bool XetlaGemmKernel<ComputeType>::run(se::gpu::GpuStreamHandle handle) {
  using gemm_policy =
      PolicyDispatcher<ComputeType,
                       GemmPolicy<ComputeType, 8, 64, 8, 16, 32, 8>,
                       GemmPolicy<ComputeType, 8, 64, 8, 16, 16, 4>,
                       GemmPolicy<ComputeType, 8, 32, 8, 16, 16, 4>,
                       GemmPolicy<ComputeType, 8, 32, 8, 16, 16, 8>,
                       GemmPolicy<ComputeType, 8, 128, 8, 16, 16, 2>,
                       GemmPolicy<ComputeType, 8, 128, 8, 16, 32, 4>,
                       GemmPolicy<ComputeType, 8, 256, 8, 16, 16, 2>,
                       GemmPolicy<ComputeType, 8, 512, 8, 16, 16, 1>,
                       GemmPolicy<ComputeType, 16, 64, 16, 16, 16, 8>,
                       GemmPolicy<ComputeType, 16, 256, 8, 16, 16, 1>,
                       GemmPolicy<ComputeType, 16, 256, 16, 16, 16, 2>,
                       GemmPolicy<ComputeType, 16, 512, 16, 16, 16, 1>,
                       GemmPolicy<ComputeType, 32, 128, 8, 16, 32, 1>,
                       GemmPolicy<ComputeType, 32, 64, 32, 16, 16, 8>,
                       GemmPolicy<ComputeType, 32, 64, 8, 16, 16, 2>,
                       GemmPolicy<ComputeType, 32, 128, 32, 16, 16, 4>,
                       GemmPolicy<ComputeType, 32, 256, 32, 16, 16, 2>,
                       GemmPolicy<ComputeType, 32, 512, 32, 16, 16, 1>,
                       GemmPolicy<ComputeType, 64, 128, 64, 16, 16, 4>,
                       GemmPolicy<ComputeType, 64, 256, 64, 16, 16, 2>,
                       GemmPolicy<ComputeType, 64, 512, 64, 16, 16, 1>,
                       GemmPolicy<ComputeType, 128, 128, 32, 32, 32, 2>,
                       GemmPolicy<ComputeType, 128, 256, 64, 16, 16, 1>,
                       GemmPolicy<ComputeType, 128, 512, 64, 32, 16, 1>,
                       GemmPolicy<ComputeType, 256, 256, 64, 32, 16, 1>,
                       GemmPolicy<ComputeType, 256, 256, 32, 64, 16, 1>,
                       GemmPolicy<ComputeType, 256, 256, 32, 64, 32, 1>,
                       GemmPolicy<ComputeType, 128, 64, 16, 16, 64, 1>,
                       GemmPolicy<ComputeType, 128, 128, 16, 32, 64, 1>,
                       GemmPolicy<ComputeType, 128, 256, 32, 32, 16, 1>>;

  int WG_M = std::get<0>(selected_policy_id_);
  int WG_N = std::get<1>(selected_policy_id_);
  int SG_M = std::get<2>(selected_policy_id_);
  int SG_N = std::get<3>(selected_policy_id_);
  int SG_K = std::get<4>(selected_policy_id_);
  int SLM_KS = std::get<5>(selected_policy_id_);
  return gemm_policy::call(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS,
                           is_b_row_major_, this, handle);
}

template class XetlaGemmKernel<sycl::half>;
template class XetlaGemmKernel<gpu::xetla::bf16>;

template <typename ComputeType>
template <int WG_M, int WG_N, int SG_M, int SG_N, int SG_K, int SLM_KS>
bool XetlaQKVGemmKernel<ComputeType>::dispatch(
    se::gpu::GpuStreamHandle handle) {
  sycl::queue q = *handle;
  if (q_out_ != nullptr && k_out_ != nullptr && v_out_ != nullptr) {
    CHECK(alpha_ == 1.0f);
    hgemm_qkv<ComputeType, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, 1, 1, 3, true>(
        q, reinterpret_cast<ComputeType*>(q_out_->data.opaque()),
        reinterpret_cast<ComputeType*>(k_out_->data.opaque()),
        reinterpret_cast<ComputeType*>(v_out_->data.opaque()),
        reinterpret_cast<ComputeType*>(a_->data.opaque()),
        reinterpret_cast<ComputeType*>(b_->data.opaque()), m_, n_, k_);
    return true;
  } else {
    LOG(ERROR) << "No mateched policy";
    return false;
  }
}

template <typename ComputeType>
bool XetlaQKVGemmKernel<ComputeType>::run(se::gpu::GpuStreamHandle handle) {
  using gemm_policy =
      PolicyDispatcher<ComputeType,
                       GemmPolicy<ComputeType, 8, 64, 8, 16, 32, 8>,
                       GemmPolicy<ComputeType, 8, 128, 8, 16, 16, 2>,
                       GemmPolicy<ComputeType, 8, 128, 8, 16, 32, 4>,
                       GemmPolicy<ComputeType, 8, 256, 8, 16, 16, 2>,
                       GemmPolicy<ComputeType, 8, 512, 8, 16, 16, 1>,
                       GemmPolicy<ComputeType, 16, 64, 16, 16, 16, 8>,
                       GemmPolicy<ComputeType, 16, 256, 8, 16, 16, 1>,
                       GemmPolicy<ComputeType, 16, 256, 16, 16, 16, 2>,
                       GemmPolicy<ComputeType, 16, 512, 16, 16, 16, 1>,
                       GemmPolicy<ComputeType, 32, 64, 32, 16, 16, 8>,
                       GemmPolicy<ComputeType, 32, 64, 8, 16, 16, 2>,
                       GemmPolicy<ComputeType, 32, 128, 32, 16, 16, 4>,
                       GemmPolicy<ComputeType, 32, 256, 32, 16, 16, 2>,
                       GemmPolicy<ComputeType, 32, 512, 32, 16, 16, 1>,
                       GemmPolicy<ComputeType, 64, 128, 64, 16, 16, 4>,
                       GemmPolicy<ComputeType, 64, 256, 64, 16, 16, 2>,
                       GemmPolicy<ComputeType, 64, 512, 64, 16, 16, 1>,
                       GemmPolicy<ComputeType, 128, 128, 32, 32, 32, 2>,
                       GemmPolicy<ComputeType, 128, 256, 64, 16, 16, 1>,
                       GemmPolicy<ComputeType, 128, 512, 64, 32, 16, 1>,
                       GemmPolicy<ComputeType, 256, 256, 64, 32, 16, 1>,
                       GemmPolicy<ComputeType, 256, 256, 32, 64, 16, 1>,
                       GemmPolicy<ComputeType, 256, 256, 32, 64, 32, 1>,
                       GemmPolicy<ComputeType, 128, 64, 16, 16, 64, 1>,
                       GemmPolicy<ComputeType, 128, 128, 16, 32, 64, 1>,
                       GemmPolicy<ComputeType, 128, 256, 32, 32, 16, 1>>;
  int WG_M = std::get<0>(selected_policy_id_);
  int WG_N = std::get<1>(selected_policy_id_);
  int SG_M = std::get<2>(selected_policy_id_);
  int SG_N = std::get<3>(selected_policy_id_);
  int SG_K = std::get<4>(selected_policy_id_);
  int SLM_KS = std::get<5>(selected_policy_id_);
  return gemm_policy::call(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS,
                           is_b_row_major_, this, handle);
}

template class XetlaQKVGemmKernel<sycl::half>;
template class XetlaQKVGemmKernel<gpu::xetla::bf16>;

}  // namespace xetla
}  // namespace gpu
