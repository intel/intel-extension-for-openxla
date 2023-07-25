/* Copyright (c) 2023 Intel Corporation

Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_GPU_ONEDNN_MATMUL_UTILS_H_
#define XLA_SERVICE_GPU_ONEDNN_MATMUL_UTILS_H_

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/scratch_allocator.h"
#include "xla/shape.h"
#include "xla/statusor.h"
#include "xla/stream_executor/blas.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

Status RunGemm(const GemmConfig& config, se::DeviceMemoryBase lhs_buffer,
               se::DeviceMemoryBase rhs_buffer, se::DeviceMemoryBase add_buffer,
               se::DeviceMemoryBase output_buffer,
               se::DeviceMemoryBase bias_buffer, se::Stream* stream,
               se::ScratchAllocator* scratch_allocator = nullptr);

namespace cublas_lt {

StatusOr<se::cuda::BlasLt::Epilogue> AsBlasLtEpilogue(
    mlir::lmhlo_gpu::CublasLtMatmulEpilogue epilogue);

class MatmulPlan {
 public:
  template <typename CublasLtMatmulMaybeF8Op,
            typename = std::enable_if<
                std::is_same<CublasLtMatmulMaybeF8Op,
                             mlir::lmhlo_gpu::CublasLtMatmulOp>::value ||
                std::is_same<CublasLtMatmulMaybeF8Op,
                             mlir::lmhlo_gpu::CublasLtMatmulF8Op>::value>>
  static StatusOr<GemmConfig> For(CublasLtMatmulMaybeF8Op op) {
    mlir::mhlo::DotDimensionNumbersAttr dot_dims = op.getDotDimensionNumbers();

    int64_t compute_precision = 0;  // Default
    if (op.getPrecisionConfig().has_value()) {
      auto precision_config = op.getPrecisionConfig();
      for (auto attr : precision_config.value()) {
        int64_t value = static_cast<int64_t>(
            attr.template cast<mlir::mhlo::PrecisionAttr>().getValue());
        if (value > compute_precision) {
          compute_precision = value;
        }
      }
    }

    Shape bias_shape;
    if (op.getBias() != nullptr) {
      bias_shape = GetShape(op.getBias());
    }
    TF_ASSIGN_OR_RETURN(
        GemmConfig config,
        GemmConfig::For(
            GetShape(op.getA()), dot_dims.getLhsBatchingDimensions(),
            dot_dims.getLhsContractingDimensions(), GetShape(op.getB()),
            dot_dims.getRhsBatchingDimensions(),
            dot_dims.getRhsContractingDimensions(), GetShape(op.getC()),
            op.getBias() == nullptr ? nullptr : &bias_shape,
            GetShape(op.getD()), op.getAlphaReal().convertToDouble(),
            op.getAlphaImag().convertToDouble(), op.getBeta().convertToDouble(),
            op.getAlgorithm(), compute_precision));

    TF_ASSIGN_OR_RETURN(se::cuda::BlasLt::Epilogue epilogue,
                        AsBlasLtEpilogue(op.getEpilogue()));
    // return From(config, epilogue);
    config.epilogue = epilogue;
    return config;
  }
};

}  // namespace cublas_lt

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_ONEDNN_MATMUL_UTILS_H_
