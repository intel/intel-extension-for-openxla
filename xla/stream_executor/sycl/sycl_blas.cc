/* Copyright (c) 2024 Intel Corporation

Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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
#include "xla/stream_executor/sycl/sycl_blas.h"

#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/stream_executor/plugin_registry.h"
#include "xla/stream_executor/sycl/sycl_platform_id.h"

namespace stream_executor {
namespace sycl {

using gpu::AsGpuStreamValue;

namespace {

oneapi::mkl::uplo SYCLBlasUpperLower(blas::UpperLower uplo) {
  switch (uplo) {
    case blas::UpperLower::kUpper:
      return oneapi::mkl::uplo::U;
    case blas::UpperLower::kLower:
      return oneapi::mkl::uplo::L;
    default:
      LOG(FATAL) << "Invalid value of blas::UpperLower.";
  }
}

oneapi::mkl::diag SYCLBlasDiagonal(blas::Diagonal diag) {
  switch (diag) {
    case blas::Diagonal::kUnit:
      return oneapi::mkl::diag::U;
    case blas::Diagonal::kNonUnit:
      return oneapi::mkl::diag::N;
    default:
      LOG(FATAL) << "Invalid value of blas::Diagonal.";
  }
}

oneapi::mkl::side SYCLBlasSide(blas::Side side) {
  switch (side) {
    case blas::Side::kLeft:
      return oneapi::mkl::side::L;
    case blas::Side::kRight:
      return oneapi::mkl::side::R;
    default:
      LOG(FATAL) << "Invalid value of blas::Side.";
  }
}

oneapi::mkl::transpose AsSyblasOperation(blas::Transpose trans) {
  switch (trans) {
    case blas::Transpose::kNoTranspose:
      return oneapi::mkl::transpose::N;
    case blas::Transpose::kTranspose:
      return oneapi::mkl::transpose::T;
    case blas::Transpose::kConjugateTranspose:
      return oneapi::mkl::transpose::C;
  }
}

}  // namespace

SYCLBlas::SYCLBlas(gpu::GpuExecutor *parent)
    : parent_(CHECK_NOTNULL(parent)), blas_(nullptr), blas_it_(nullptr) {}

bool SYCLBlas::Init() { return true; }

SYCLBlas::~SYCLBlas() {}

bool SYCLBlas::SetStream(Stream *stream) { return true; }

syclStream_t SYCLBlas::SYCLStream(Stream *stream) {
  CHECK(stream != nullptr);
  CHECK(AsGpuStreamValue(stream) != nullptr);
  return AsGpuStreamValue(stream);
}

bool SYCLBlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64_t m, uint64 n,
                          float alpha, const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *b, int ldb) {
  auto gpu_stream = stream_executor::gpu::AsGpuStreamValue(stream);
  oneapi::mkl::blas::trsm(*gpu_stream, SYCLBlasSide(side),
                          SYCLBlasUpperLower(uplo), AsSyblasOperation(transa),
                          SYCLBlasDiagonal(diag), m, n, &alpha,
                          static_cast<const float *>(a.opaque()), lda,
                          static_cast<float *>(b->opaque()), ldb);
  return true;
}

bool SYCLBlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64_t m, uint64 n,
                          double alpha, const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *b, int ldb) {
  auto gpu_stream = stream_executor::gpu::AsGpuStreamValue(stream);
  oneapi::mkl::blas::trsm(*gpu_stream, SYCLBlasSide(side),
                          SYCLBlasUpperLower(uplo), AsSyblasOperation(transa),
                          SYCLBlasDiagonal(diag), m, n, &alpha,
                          static_cast<const double *>(a.opaque()), lda,
                          static_cast<double *>(b->opaque()), ldb);
  return true;
}

bool SYCLBlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64_t m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          DeviceMemory<std::complex<float>> *b, int ldb) {
  auto gpu_stream = stream_executor::gpu::AsGpuStreamValue(stream);
  oneapi::mkl::blas::trsm(
      *gpu_stream, SYCLBlasSide(side), SYCLBlasUpperLower(uplo),
      AsSyblasOperation(transa), SYCLBlasDiagonal(diag), m, n, &alpha,
      static_cast<const std::complex<float> *>(a.opaque()), lda,
      static_cast<std::complex<float> *>(b->opaque()), ldb);
  return true;
}

bool SYCLBlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64_t m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          DeviceMemory<std::complex<double>> *b, int ldb) {
  auto gpu_stream = stream_executor::gpu::AsGpuStreamValue(stream);
  oneapi::mkl::blas::trsm(
      *gpu_stream, SYCLBlasSide(side), SYCLBlasUpperLower(uplo),
      AsSyblasOperation(transa), SYCLBlasDiagonal(diag), m, n, &alpha,
      static_cast<const std::complex<double> *>(a.opaque()), lda,
      static_cast<std::complex<double> *>(b->opaque()), ldb);
  return true;
}

bool SYCLBlas::DoBlasTrsmBatched(Stream *stream, blas::Side side,
                                 blas::UpperLower uplo, blas::Transpose transa,
                                 blas::Diagonal diag, uint64_t m, uint64 n,
                                 float alpha, const DeviceMemory<float *> &as,
                                 int lda, DeviceMemory<float *> *bs, int ldb,
                                 int batch_count) {
  auto gpu_stream = stream_executor::gpu::AsGpuStreamValue(stream);
  const int stride_a = lda * lda;
  const int stride_b = m * n;
  oneapi::mkl::blas::trsm_batch(
      *gpu_stream, SYCLBlasSide(side), SYCLBlasUpperLower(uplo),
      AsSyblasOperation(transa), SYCLBlasDiagonal(diag), m, n, &alpha,
      static_cast<const float *>(as.opaque()), lda, stride_a,
      static_cast<float *>(bs->opaque()), ldb, stride_b, batch_count);
  return true;
}

bool SYCLBlas::DoBlasTrsmBatched(Stream *stream, blas::Side side,
                                 blas::UpperLower uplo, blas::Transpose transa,
                                 blas::Diagonal diag, uint64_t m, uint64 n,
                                 double alpha, const DeviceMemory<double *> &as,
                                 int lda, DeviceMemory<double *> *bs, int ldb,
                                 int batch_count) {
  auto gpu_stream = stream_executor::gpu::AsGpuStreamValue(stream);
  const int stride_a = lda * lda;
  const int stride_b = m * n;
  oneapi::mkl::blas::trsm_batch(
      *gpu_stream, SYCLBlasSide(side), SYCLBlasUpperLower(uplo),
      AsSyblasOperation(transa), SYCLBlasDiagonal(diag), m, n, &alpha,
      static_cast<const double *>(as.opaque()), lda, stride_a,
      static_cast<double *>(bs->opaque()), ldb, stride_b, batch_count);
  return true;
}

bool SYCLBlas::DoBlasTrsmBatched(Stream *stream, blas::Side side,
                                 blas::UpperLower uplo, blas::Transpose transa,
                                 blas::Diagonal diag, uint64_t m, uint64 n,
                                 std::complex<float> alpha,
                                 const DeviceMemory<std::complex<float> *> &as,
                                 int lda,
                                 DeviceMemory<std::complex<float> *> *bs,
                                 int ldb, int batch_count) {
  auto gpu_stream = stream_executor::gpu::AsGpuStreamValue(stream);
  const int stride_a = lda * lda;
  const int stride_b = m * n;
  oneapi::mkl::blas::trsm_batch(
      *gpu_stream, SYCLBlasSide(side), SYCLBlasUpperLower(uplo),
      AsSyblasOperation(transa), SYCLBlasDiagonal(diag), m, n, &alpha,
      static_cast<const std::complex<float> *>(as.opaque()), lda, stride_a,
      static_cast<std::complex<float> *>(bs->opaque()), ldb, stride_b,
      batch_count);
  return true;
}

bool SYCLBlas::DoBlasTrsmBatched(Stream *stream, blas::Side side,
                                 blas::UpperLower uplo, blas::Transpose transa,
                                 blas::Diagonal diag, uint64_t m, uint64 n,
                                 std::complex<double> alpha,
                                 const DeviceMemory<std::complex<double> *> &as,
                                 int lda,
                                 DeviceMemory<std::complex<double> *> *bs,
                                 int ldb, int batch_count) {
  auto gpu_stream = stream_executor::gpu::AsGpuStreamValue(stream);
  const int stride_a = lda * lda;
  const int stride_b = m * n;
  oneapi::mkl::blas::trsm_batch(
      *gpu_stream, SYCLBlasSide(side), SYCLBlasUpperLower(uplo),
      AsSyblasOperation(transa), SYCLBlasDiagonal(diag), m, n, &alpha,
      static_cast<const std::complex<double> *>(as.opaque()), lda, stride_a,
      static_cast<std::complex<double> *>(bs->opaque()), ldb, stride_b,
      batch_count);
  return true;
}

// Undefinition
bool SYCLBlas::DoBlasAxpy(Stream *stream, uint64_t elem_count, float alpha,
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *y, int incy) {
  return true;
}
bool SYCLBlas::DoBlasCopy(Stream *stream, uint64_t elem_count,
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *y, int incy) {
  return true;
}
bool SYCLBlas::DoBlasScal(Stream *stream, uint64_t elem_count, float alpha,
                          DeviceMemory<float> *x, int incx) {
  return true;
}
bool SYCLBlas::DoBlasScal(Stream *stream, uint64_t elem_count, double alpha,
                          DeviceMemory<double> *x, int incx) {
  return true;
}
bool SYCLBlas::DoBlasScal(Stream *stream, uint64_t elem_count, float alpha,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  return true;
}
bool SYCLBlas::DoBlasScal(Stream *stream, uint64_t elem_count, double alpha,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  return true;
}
bool SYCLBlas::DoBlasScal(Stream *stream, uint64_t elem_count,
                          std::complex<float> alpha,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  return true;
}
bool SYCLBlas::DoBlasScal(Stream *stream, uint64_t elem_count,
                          std::complex<double> alpha,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  return true;
}
bool SYCLBlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64_t m,
                          uint64 n, float alpha, const DeviceMemory<float> &a,
                          int lda, const DeviceMemory<float> &x, int incx,
                          float beta, DeviceMemory<float> *y, int incy) {
  return true;
}
bool SYCLBlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64_t m,
                          uint64 n, double alpha, const DeviceMemory<double> &a,
                          int lda, const DeviceMemory<double> &x, int incx,
                          double beta, DeviceMemory<double> *y, int incy) {
  return true;
}
bool SYCLBlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64_t m,
                          uint64 n, std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  return true;
}
bool SYCLBlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64_t m,
                          uint64 n, std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  return true;
}
bool SYCLBlas::DoBlasSbmv(Stream *stream, blas::UpperLower uplo, uint64_t n,
                          uint64 k, float alpha, const DeviceMemory<float> &a,
                          int lda, const DeviceMemory<float> &x, int incx,
                          float beta, DeviceMemory<float> *y, int incy) {
  return true;
}
absl::Status SYCLBlas::DoBlasGemm(Stream *stream, blas::Transpose transa,
                                 blas::Transpose transb, uint64_t m, uint64 n,
                                 uint64 k, blas::DataType dtype,
                                 const void *alpha, const DeviceMemoryBase &a,
                                 int lda, const DeviceMemoryBase &b, int ldb,
                                 const void *beta, DeviceMemoryBase *c, int ldc,
                                 const NumericOptions &numeric_options,
                                 blas::CallContext context) {
  return absl::UnimplementedError("Not implemented for SYCL");
}
bool SYCLBlas::GetBlasGemmAlgorithms(
    Stream *stream, const gpu::MatrixDescriptor &,
    const gpu::MatrixDescriptor &, gpu::OutputMatrixDescriptor *, const void *,
    const void *, std::vector<blas::AlgorithmType> *out_algorithms) {
  return true;
}
absl::Status SYCLBlas::DoBlasGemmWithAlgorithm(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64 n, uint64 k, const void *alpha, const DeviceMemoryBase &a,
    blas::DataType type_a, int lda, const DeviceMemoryBase &b,
    blas::DataType type_b, int ldb, const void *beta, DeviceMemoryBase *c,
    blas::DataType type_c, int ldc, blas::ComputationType computation_type,
    blas::AlgorithmType algorithm, const NumericOptions &numeric_options,
    blas::ProfileResult *output_profile_result, blas::CallContext context) {
  return absl::UnimplementedError("Not implemented for SYCL");
}
bool SYCLBlas::DoBlasGemmBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64 n, uint64 k, float alpha, DeviceMemorySlice<Eigen::half> a, int lda,
    DeviceMemorySlice<Eigen::half> b, int ldb, float beta,
    DeviceMemorySlice<Eigen::half> c, int ldc, int batch_count,
    const NumericOptions &numeric_options, ScratchAllocator *scratch_allocator,
    blas::CallContext context) {
  return true;
}
bool SYCLBlas::DoBlasGemmBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64 n, uint64 k, float alpha, DeviceMemorySlice<Eigen::bfloat16> a,
    int lda, DeviceMemorySlice<Eigen::bfloat16> b, int ldb, float beta,
    DeviceMemorySlice<Eigen::bfloat16> c, int ldc, int batch_count,
    const NumericOptions &numeric_options, ScratchAllocator *scratch_allocator,
    blas::CallContext context) {
  return true;
}
bool SYCLBlas::DoBlasGemmBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64 n, uint64 k, float alpha, DeviceMemorySlice<float> a, int lda,
    DeviceMemorySlice<float> b, int ldb, float beta, DeviceMemorySlice<float> c,
    int ldc, int batch_count, const NumericOptions &numeric_options,
    ScratchAllocator *scratch_allocator, blas::CallContext context) {
  return true;
}
bool SYCLBlas::DoBlasGemmBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64 n, uint64 k, double alpha, DeviceMemorySlice<double> a, int lda,
    DeviceMemorySlice<double> b, int ldb, double beta,
    DeviceMemorySlice<double> c, int ldc, int batch_count,
    const NumericOptions &numeric_options, ScratchAllocator *scratch_allocator,
    blas::CallContext context) {
  return true;
}
bool SYCLBlas::DoBlasGemmBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64 n, uint64 k, std::complex<float> alpha,
    DeviceMemorySlice<std::complex<float>> a, int lda,
    DeviceMemorySlice<std::complex<float>> b, int ldb, std::complex<float> beta,
    DeviceMemorySlice<std::complex<float>> c, int ldc, int batch_count,
    const NumericOptions &numeric_options, ScratchAllocator *scratch_allocator,
    blas::CallContext context) {
  return true;
}
bool SYCLBlas::DoBlasGemmBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64 n, uint64 k, std::complex<double> alpha,
    DeviceMemorySlice<std::complex<double>> a, int lda,
    DeviceMemorySlice<std::complex<double>> b, int ldb,
    std::complex<double> beta, DeviceMemorySlice<std::complex<double>> c,
    int ldc, int batch_count, const NumericOptions &numeric_options,
    ScratchAllocator *scratch_allocator, blas::CallContext context) {
  return true;
}
absl::Status SYCLBlas::DoBlasGemmStridedBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64 n, uint64 k, blas::DataType dtype, const void *alpha,
    const DeviceMemoryBase &a, int lda, int64_t stride_a,
    const DeviceMemoryBase &b, int ldb, int64_t stride_b, const void *beta,
    DeviceMemoryBase *c, int ldc, int64_t stride_c, int batch_count,
    const NumericOptions &numeric_options, blas::CallContext context) {
  return absl::UnimplementedError("Not implemented for SYCL");
}
absl::Status SYCLBlas::DoBlasGemmStridedBatchedWithAlgorithm(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64 n, uint64 k, const void *alpha, const DeviceMemoryBase &a,
    blas::DataType type_a, int lda, int64_t stride_a, const DeviceMemoryBase &b,
    blas::DataType type_b, int ldb, int64_t stride_b, const void *beta,
    DeviceMemoryBase *c, blas::DataType type_c, int ldc, int64_t stride_c,
    int batch_count, blas::ComputationType computation_type,
    blas::AlgorithmType algorithm, const NumericOptions &numeric_options,
    blas::ProfileResult *output_profile_result, blas::CallContext context) {
  return absl::UnimplementedError("Not implemented for SYCL");
}
absl::Status SYCLBlas::GetVersion(std::string *version) {
  return absl::UnimplementedError("Not implemented for SYCL");
}
// Undefinition

void initialize_syclblas() {
  absl::Status status =
      PluginRegistry::Instance()->RegisterFactory<PluginRegistry::BlasFactory>(
          kSyclPlatformId, "syBLAS",
          [](::stream_executor::StreamExecutor* parent) -> blas::BlasSupport * {
            gpu::GpuExecutor *sycl_executor =
                dynamic_cast<gpu::GpuExecutor *>(parent);
            if (sycl_executor == nullptr) {
              LOG(ERROR)
                  << "Attempting to initialize an instance of the syBLAS "
                  << "support library with a non-SYCL StreamExecutor";
              return nullptr;
            }

            SYCLBlas *blas = new SYCLBlas(sycl_executor);
            return blas;
          });

  if (!status.ok()) {
    LOG(ERROR) << "Unable to register syBLAS factory: " << status.message();
  }
}

}  // namespace sycl
}  // namespace stream_executor

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(register_syclblas, {
  stream_executor::sycl::initialize_syclblas();
});

