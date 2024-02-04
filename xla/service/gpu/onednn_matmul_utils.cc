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

#include "xla/service/gpu/onednn_matmul_utils.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include <xetla.hpp>

#include "absl/algorithm/container.h"
#include "absl/types/span.h"
#include "dnnl.hpp"       // NOLINT(build/include_subdir)
#include "dnnl_sycl.hpp"  // NOLINT(build/include_subdir)
#include "tsl/framework/numeric_types.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/types.h"
#include "tsl/util/env_var.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/mlir_hlo/lhlo_gpu/IR/lhlo_gpu_ops.h"
#include "xla/service/gpu/matrix_descriptor.h"
#include "xla/service/gpu/xetla/gemm/gemm.h"
#include "xla/service/onednn_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/statusor.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/sycl/hw_info.h"
#include "xla/stream_executor/sycl/sycl_stream.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

/// Return oneDNN data type (memory::data_type) for input type T
///
/// @input None
/// @return dnnl::memory::data_type corresponding to type T
template <typename T>
inline dnnl::memory::data_type OneDnnType();

/// Instantiation for float type. Add similar instantiations for other
/// type if needed.
template <>
inline dnnl::memory::data_type OneDnnType<float>() {
  return dnnl::memory::data_type::f32;
}

template <>
inline dnnl::memory::data_type OneDnnType<double>() {
  return dnnl::memory::data_type::f64;
}

template <>
inline dnnl::memory::data_type OneDnnType<sycl::half>() {
  return dnnl::memory::data_type::f16;
}

template <>
inline dnnl::memory::data_type OneDnnType<tsl::quint8>() {
  return dnnl::memory::data_type::u8;
}

template <>
inline dnnl::memory::data_type OneDnnType<tsl::uint8>() {
  return dnnl::memory::data_type::u8;
}

template <>
inline dnnl::memory::data_type OneDnnType<tsl::qint8>() {
  return dnnl::memory::data_type::s8;
}

template <>
inline dnnl::memory::data_type OneDnnType<tsl::qint32>() {
  return dnnl::memory::data_type::s32;
}

template <>
inline dnnl::memory::data_type OneDnnType<::gpu::xetla::bf16>() {
  return dnnl::memory::data_type::bf16;
}

namespace {

MatrixDescriptor GetMatrixDesc(const MatrixLayout& layout,
                               se::DeviceMemoryBase data) {
  bool transpose = layout.order == MatrixLayout::Order::kColumnMajor;
  return MatrixDescriptor{
      data,
      transpose ? se::blas::Transpose::kTranspose
                : se::blas::Transpose::kNoTranspose,
      transpose ? layout.num_cols : layout.num_rows,
      transpose ? layout.num_rows : layout.num_cols,
      *layout.batch_stride,
      *layout.leading_dim_stride,
  };
}

struct OneDnnMatMulParams {
  dnnl::memory::dims a_dims;
  dnnl::memory::dims b_dims;
  dnnl::memory::dims c_dims;
  dnnl::memory::dims bias_dims;
  dnnl::memory::dims a_strides;
  dnnl::memory::dims b_strides;
  dnnl::memory::dims c_strides;
  dnnl::memory::dims bias_strides;

  OneDnnMatMulParams(dnnl::memory::dims a_dims, dnnl::memory::dims b_dims,
                     dnnl::memory::dims c_dims, dnnl::memory::dims bias_dims,
                     dnnl::memory::dims a_strides, dnnl::memory::dims b_strides,
                     dnnl::memory::dims c_strides,
                     dnnl::memory::dims bias_strides)
      : a_dims(std::move(a_dims)),
        b_dims(std::move(b_dims)),
        c_dims(std::move(c_dims)),
        bias_dims(std::move(bias_dims)),
        a_strides(std::move(a_strides)),
        b_strides(std::move(b_strides)),
        c_strides(std::move(c_strides)),
        bias_strides(std::move(bias_strides)) {}
};

template <typename InputT>
std::enable_if_t<std::is_same_v<InputT, ::gpu::xetla::bf16> ||
                     std::is_same_v<InputT, sycl::half>,
                 StatusOr<bool>>
RunXetlaGemm(se::gpu::GpuStreamHandle handle, const MatrixDescriptor& lhs,
             const MatrixDescriptor& rhs, const MatrixDescriptor& c,
             const MatrixDescriptor& out, se::DeviceMemoryBase bias,
             se::gpu::BlasLt::Epilogue epilogue, float beta) {
  void* bias_data = const_cast<void*>(bias.opaque());
  void* c_data = const_cast<void*>(c.data.opaque());
  switch (epilogue) {
    case se::gpu::BlasLt::Epilogue::kDefault: {
      auto policy = ::gpu::xetla::XetlaGemmKernel<InputT>()
                        .add_matrix_c(out)
                        .add_matrix_a(lhs)
                        .add_matrix_b(rhs)
                        .build();
      if (fabs(beta) - 0.0f > 1e-6) {
        if (fabs(beta) - 1.0f < 1e-6) {
          policy
              .add_epilogue(
                  c_data,
                  ::gpu::xetla::XetlaGemmKernel<InputT>::EpilogueType::RES_ADD)
              .build();
        } else {
          return true;
        }
      }
      if (policy.fallback() == false) {
        policy.run(handle);
      }
      return policy.fallback();
    }
    case se::gpu::BlasLt::Epilogue::kBias: {
      auto policy =
          ::gpu::xetla::XetlaGemmKernel<InputT>()
              .add_matrix_c(out)
              .add_matrix_a(lhs)
              .add_matrix_b(rhs)
              .add_epilogue(
                  bias_data,
                  ::gpu::xetla::XetlaGemmKernel<InputT>::EpilogueType::BIAS)
              .build();
      if (fabs(beta) - 0.0f > 1e-6) {
        policy
            .add_epilogue(
                c_data,
                ::gpu::xetla::XetlaGemmKernel<InputT>::EpilogueType::RES_ADD,
                beta)
            .build();
      }
      if (policy.fallback() == false) {
        policy.run(handle);
      }
      return policy.fallback();
    }
    case se::gpu::BlasLt::Epilogue::kGELU: {
      auto policy =
          ::gpu::xetla::XetlaGemmKernel<InputT>()
              .add_matrix_c(out)
              .add_matrix_a(lhs)
              .add_matrix_b(rhs)
              .add_epilogue(
                  nullptr,
                  ::gpu::xetla::XetlaGemmKernel<InputT>::EpilogueType::GELU)
              .build();
      if (policy.fallback() == false) {
        policy.run(handle);
      }
      return policy.fallback();
    }
    case se::gpu::BlasLt::Epilogue::kBiasThenGELU: {
      auto policy =
          ::gpu::xetla::XetlaGemmKernel<InputT>()
              .add_matrix_c(out)
              .add_matrix_a(lhs)
              .add_matrix_b(rhs)
              .add_epilogue(
                  bias_data,
                  ::gpu::xetla::XetlaGemmKernel<InputT>::EpilogueType::BIAS)
              .add_epilogue(
                  nullptr,
                  ::gpu::xetla::XetlaGemmKernel<InputT>::EpilogueType::GELU)
              .build();
      if (policy.fallback() == false) {
        policy.run(handle);
      }
      return policy.fallback();
    }
    case se::gpu::BlasLt::Epilogue::kReLU:
    case se::gpu::BlasLt::Epilogue::kBiasThenReLU:
      return true;
    default:
      return InternalError("Unsupported Activation mode");
  }
}

template <typename InputT>
std::enable_if_t<std::is_same_v<InputT, float>, StatusOr<bool>> RunXetlaGemm(
    se::gpu::GpuStreamHandle handle, const MatrixDescriptor& lhs,
    const MatrixDescriptor& rhs, const MatrixDescriptor& c,
    const MatrixDescriptor& out, se::DeviceMemoryBase bias,
    se::gpu::BlasLt::Epilogue epilogue, float beta) {
  return InternalError("Unsupported Datatype in XeTLA");
}

std::unique_ptr<OneDnnMatMulParams> CreateMatMulParams(
    int64_t batch_size, const MatrixDescriptor& lhs,
    const MatrixDescriptor& rhs, const MatrixDescriptor& out) {
  dnnl::memory::dims lhs_dims{batch_size, lhs.num_rows, lhs.num_cols};
  dnnl::memory::dims rhs_dims{batch_size, rhs.num_rows, rhs.num_cols};
  dnnl::memory::dims out_dims{batch_size, out.num_rows, out.num_cols};

  auto lhs_strides =
      dnnl::memory::dims{lhs.batch_stride, lhs.leading_dim_stride, 1};
  auto rhs_strides =
      dnnl::memory::dims{rhs.batch_stride, rhs.leading_dim_stride, 1};
  auto out_strides =
      dnnl::memory::dims{out.batch_stride, out.leading_dim_stride, 1};
  int idx_last = 2;
  int idx_2nd_last = 1;

  // dst(m,n) = \sigma{src(m,k) * weights(k, n)}
  // lhs_strides holds the strides for each dim, say {24, 12, 4, 1} for
  // src_tensor {1, 2, 3, 4} if adj_x_ is false.
  // If adj_x_ is true, swap the innermost two dims of lhs_strides
  // to {24, 12, 1, 4}, just like set memory::format_tag::abdc
  if (lhs.transpose == se::blas::Transpose::kTranspose) {
    std::swap(lhs_dims[idx_last], lhs_dims[idx_2nd_last]);
    std::swap(lhs_strides[idx_last], lhs_strides[idx_2nd_last]);
  }
  if (rhs.transpose == se::blas::Transpose::kTranspose) {
    std::swap(rhs_dims[idx_last], rhs_dims[idx_2nd_last]);
    std::swap(rhs_strides[idx_last], rhs_strides[idx_2nd_last]);
  }

  dnnl::memory::dims bias_dims(rhs_dims.size(), 1);
  bias_dims[rhs_dims.size() - 1] = rhs_dims[rhs_dims.size() - 1];
  auto bias_strides = CalculateTFStrides(bias_dims);

  return absl::make_unique<OneDnnMatMulParams>(
      lhs_dims, rhs_dims, out_dims, bias_dims, lhs_strides, rhs_strides,
      out_strides, bias_strides);
}

template <typename InputT>
Status DoGemm(int64_t batch_size, int64_t m, int64_t n, int64_t k,
              const MatrixDescriptor& lhs, const MatrixDescriptor& rhs,
              const MatrixDescriptor& c, const MatrixDescriptor& output,
              se::DeviceMemoryBase bias, float alpha, float beta,
              se::gpu::BlasLt::Epilogue epilogue, se::Stream* stream,
              se::ScratchAllocator* scratch_allocator,
              se::blas::ComputePrecision compute_precision) {
  CHECK(output.transpose == se::blas::Transpose::kNoTranspose);
  se::gpu::GpuStreamHandle stream_handle =
      stream_executor::gpu::AsGpuStreamValue(stream);
  void* lhs_data = const_cast<void*>(lhs.data.opaque());
  void* rhs_data = const_cast<void*>(rhs.data.opaque());
  void* c_data = const_cast<void*>(c.data.opaque());
  void* out_data = const_cast<void*>(output.data.opaque());
  void* bias_data = const_cast<void*>(bias.opaque());

  VLOG(2) << "lhs: " << batch_size << " " << lhs.num_rows << " "
          << lhs.num_cols;
  VLOG(2) << "rhs: " << batch_size << " " << rhs.num_rows << " "
          << rhs.num_cols;
  VLOG(2) << "out: " << batch_size << " " << output.num_rows << " "
          << output.num_cols;
  VLOG(2) << "lhs stride: " << lhs.batch_stride << " " << lhs.leading_dim_stride
          << " " << 1;
  VLOG(2) << "rhs stride: " << rhs.batch_stride << " " << rhs.leading_dim_stride
          << " " << 1;
  VLOG(2) << "out stride: " << output.batch_stride << " "
          << output.leading_dim_stride << " " << 1;
  VLOG(2) << "lhs trans: " << TransposeString(lhs.transpose);
  VLOG(2) << "rhs trans: " << TransposeString(rhs.transpose);

  bool flag = false;
  tsl::ReadBoolFromEnvVar("XETLA_GEMM", false, &flag);
  bool xetla_support = flag && IsXetlaHardwareSupport() && (batch_size == 1) &&
                       (fabs(alpha - 1.0f) < 1e-6);
  if (xetla_support && (!std::is_same_v<InputT, float>)) {
    TF_ASSIGN_OR_RETURN(bool fallback,
                        RunXetlaGemm<InputT>(stream_handle, lhs, rhs, c, output,
                                             bias, epilogue, beta));
    if (!fallback) return OkStatus();
  }
  auto params = CreateMatMulParams(batch_size, lhs, rhs, output);

  auto src_md = dnnl::memory::desc(params->a_dims, OneDnnType<InputT>(),
                                   params->a_strides);
  auto weights_md = dnnl::memory::desc(params->b_dims, OneDnnType<InputT>(),
                                       params->b_strides);
  auto dst_md = dnnl::memory::desc(params->c_dims, OneDnnType<InputT>(),
                                   params->c_strides);
  auto bias_md =
      bias_data ? dnnl::memory::desc(params->bias_dims, OneDnnType<InputT>(),
                                     params->bias_strides)
                : dnnl::memory::desc();

  auto dnnl_engine = FindOrCreateEngine(stream_handle);
  dnnl::primitive_attr post_ops_attr;
  post_ops_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  // Set fp32 mode.
  dnnl::fpmath_mode fp32_math_mode = GetFP32MathMode();
  if (std::is_same<InputT, float>::value) {
    post_ops_attr.set_fpmath_mode(fp32_math_mode);
  }

  dnnl::post_ops post_ops = dnnl::post_ops();
  // C = activation(MatMul(x, w, bias) + beta * C)
  //   po.append_sum(beta)
  //   po.append_eltwise(dnnl::algorithm::activation, 1, 0);
  CHECK(fabs(alpha - 1.0f) < 1e-6);
  if (c_data && fabs(beta - 0.0f) > 1e-6) post_ops.append_sum(beta);
  switch (epilogue) {
    case se::gpu::BlasLt::Epilogue::kReLU:
    case se::gpu::BlasLt::Epilogue::kBiasThenReLU:
      post_ops.append_eltwise(dnnl::algorithm::eltwise_relu, 0, 0);
      break;
    case se::gpu::BlasLt::Epilogue::kGELU:
    case se::gpu::BlasLt::Epilogue::kBiasThenGELU:
      post_ops.append_eltwise(dnnl::algorithm::eltwise_gelu_tanh, 0, 0);
      break;
    case se::gpu::BlasLt::Epilogue::kDefault:
    case se::gpu::BlasLt::Epilogue::kBias:
      break;
    default:
      return InternalError("Unsupported Activation mode");
  }
  post_ops_attr.set_post_ops(post_ops);

  auto matmul_pd =
      bias_data
          ? std::make_shared<dnnl::matmul::primitive_desc>(
                dnnl_engine, src_md, weights_md, bias_md, dst_md, post_ops_attr)
          : std::make_shared<dnnl::matmul::primitive_desc>(
                dnnl_engine, src_md, weights_md, dst_md, post_ops_attr);
  std::unordered_map<int, dnnl::memory> fwd_primitive_args;

  size_t scratchpad_size = matmul_pd->scratchpad_desc().get_size();
  void* workspace;
  TF_RETURN_IF_ERROR(
      AllocateWorkspace(&workspace, scratch_allocator, scratchpad_size));

  auto scratchpad_mem =
      dnnl::memory(matmul_pd->scratchpad_desc(), dnnl_engine, workspace);

  auto matmul_primitive = dnnl::matmul(*matmul_pd);

  auto dnnl_stream = dnnl::sycl_interop::make_stream(
      dnnl_engine, *(stream_executor::gpu::AsGpuStreamValue(stream)));
  auto src_mem = CreateDnnlMemory(src_md, dnnl_engine, lhs_data);

  auto wei_mem = CreateDnnlMemory(weights_md, dnnl_engine, rhs_data);
  auto dst_mem = CreateDnnlMemory(dst_md, dnnl_engine, out_data);
  fwd_primitive_args.emplace(DNNL_ARG_SRC, src_mem);
  fwd_primitive_args.emplace(DNNL_ARG_WEIGHTS, wei_mem);
  fwd_primitive_args.emplace(DNNL_ARG_DST, dst_mem);
  fwd_primitive_args.emplace(DNNL_ARG_SCRATCHPAD, scratchpad_mem);
  if (bias_data) {
    auto bias_mem = CreateDnnlMemory(bias_md, dnnl_engine, bias_data);
    fwd_primitive_args.emplace(DNNL_ARG_BIAS, bias_mem);
  }
  matmul_primitive.execute(dnnl_stream, fwd_primitive_args);
  return OkStatus();
}

void TransposeMatrixDesc(MatrixDescriptor& matrix_desc) {
  matrix_desc.transpose =
      (matrix_desc.transpose == se::blas::Transpose::kNoTranspose)
          ? se::blas::Transpose::kTranspose
          : se::blas::Transpose::kNoTranspose;
}

void MakeBlasGemmCompatible(MatrixDescriptor& lhs, MatrixDescriptor& rhs,
                            MatrixDescriptor& output) {
  // BLAS GeMM doesn't support transposed output, but we can use the identity:
  // C^T = (A @ B)^T = B^T @ A^T.
  if (output.transpose == se::blas::Transpose::kTranspose) {
    std::swap(lhs, rhs);
    TransposeMatrixDesc(lhs);
    TransposeMatrixDesc(rhs);
    TransposeMatrixDesc(output);
  }
}

void MakeBlasGemmCompatible(MatrixDescriptor& lhs, MatrixDescriptor& rhs,
                            MatrixDescriptor& c, MatrixDescriptor& output) {
  // BLAS GeMM doesn't support transposed output, but we can use the identity:
  // C^T = (A @ B)^T = B^T @ A^T.
  if (output.transpose == se::blas::Transpose::kTranspose) {
    std::swap(lhs, rhs);
    TransposeMatrixDesc(lhs);
    TransposeMatrixDesc(rhs);
    TransposeMatrixDesc(output);
    TransposeMatrixDesc(c);
  }
}
}  // namespace

Status RunGemm(const GemmConfig& config, se::DeviceMemoryBase lhs_buffer,
               se::DeviceMemoryBase rhs_buffer, se::DeviceMemoryBase c_buffer,
               se::DeviceMemoryBase output_buffer,
               se::DeviceMemoryBase bias_buffer, se::Stream* stream,
               se::gpu::BlasLt::Epilogue epilogue,
               se::ScratchAllocator* scratch_allocator) {
  VLOG(2) << "Executing a GemmThunk";

  auto lhs_layout = MatrixLayout{config.lhs_layout},
       rhs_layout = MatrixLayout{config.rhs_layout},
       output_layout = MatrixLayout{config.output_layout},
       c_layout = MatrixLayout{config.c_layout};

  int64_t m = output_layout.num_rows;
  int64_t n = output_layout.num_cols;
  int64_t k = lhs_layout.num_cols;
  MatrixDescriptor lhs = GetMatrixDesc(lhs_layout, lhs_buffer);
  MatrixDescriptor rhs = GetMatrixDesc(rhs_layout, rhs_buffer);
  MatrixDescriptor c = GetMatrixDesc(c_layout, c_buffer);
  MatrixDescriptor output = GetMatrixDesc(output_layout, output_buffer);
  int64_t batch_size = output_layout.batch_size;
  MakeBlasGemmCompatible(lhs, rhs, c, output);

  if ((output_layout.dtype == F16 || output_layout.dtype == BF16 ||
       output_layout.dtype == F32 || output_layout.dtype == F64 ||
       output_layout.dtype == C64 || output_layout.dtype == C128) &&
      (lhs_layout.dtype != output_layout.dtype ||
       rhs_layout.dtype != output_layout.dtype)) {
    return InternalError(
        "GEMM lhs type(%s) and rhs type(%s) must match output type(%s)",
        primitive_util::LowercasePrimitiveTypeName(lhs_layout.dtype),
        primitive_util::LowercasePrimitiveTypeName(rhs_layout.dtype),
        primitive_util::LowercasePrimitiveTypeName(output_layout.dtype));
  }

  switch (output_layout.dtype) {
    case F16:
      return DoGemm<sycl::half>(batch_size, m, n, k, lhs, rhs, c, output,
                                bias_buffer, config.alpha.real(), config.beta,
                                epilogue, stream, scratch_allocator,
                                config.compute_precision);
    case BF16:
      return DoGemm<::gpu::xetla::bf16>(
          batch_size, m, n, k, lhs, rhs, c, output, bias_buffer,
          config.alpha.real(), config.beta, epilogue, stream, scratch_allocator,
          config.compute_precision);
    case F32:
      return DoGemm<float>(batch_size, m, n, k, lhs, rhs, c, output,
                           bias_buffer, config.alpha.real(), config.beta,
                           epilogue, stream, scratch_allocator,
                           config.compute_precision);
    case S32:
    case F64:
    case C64:
    case C128:
    default:
      return InternalError(
          "Unexpected GEMM dtype: %s",
          primitive_util::LowercasePrimitiveTypeName(output_layout.dtype));
  }
}

// namespace cublas_lt {
// StatusOr<se::gpu::BlasLt::Epilogue> AsBlasLtEpilogue(
//     mlir::lmhlo_gpu::CublasLtMatmulEpilogue epilogue) {
//   switch (epilogue) {
//     case mlir::lmhlo_gpu::CublasLtMatmulEpilogue::Default:
//       return se::gpu::BlasLt::Epilogue::kDefault;
//     case mlir::lmhlo_gpu::CublasLtMatmulEpilogue::Relu:
//       return se::gpu::BlasLt::Epilogue::kReLU;
//     case mlir::lmhlo_gpu::CublasLtMatmulEpilogue::Gelu:
//       return se::gpu::BlasLt::Epilogue::kGELU;
//     case mlir::lmhlo_gpu::CublasLtMatmulEpilogue::GeluAux:
//       return se::gpu::BlasLt::Epilogue::kGELUWithAux;
//     case mlir::lmhlo_gpu::CublasLtMatmulEpilogue::Bias:
//       return se::gpu::BlasLt::Epilogue::kBias;
//     case mlir::lmhlo_gpu::CublasLtMatmulEpilogue::BiasRelu:
//       return se::gpu::BlasLt::Epilogue::kBiasThenReLU;
//     case mlir::lmhlo_gpu::CublasLtMatmulEpilogue::BiasGelu:
//       return se::gpu::BlasLt::Epilogue::kBiasThenGELU;
//     case mlir::lmhlo_gpu::CublasLtMatmulEpilogue::BiasGeluAux:
//       return se::gpu::BlasLt::Epilogue::kBiasThenGELUWithAux;
//   }
//   return InternalError("unexpected epilogue value");
// }
// }  // namespace cublas_lt

}  // namespace gpu
}  // namespace xla
