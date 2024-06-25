/* Copyright (c) 2024 Intel Corporation

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

#include "xla/service/gpu/sycl_onednn.h"

#include <regex>

namespace xla {
namespace gpu {

absl::Status GetBackendDict(
    const ffi::Dictionary& dict,
    absl::flat_hash_map<std::string, std::string>& backend_dict,
    std::string& config_name) {
  std::string backend_config_str =
      std::string(*dict.get<std::string_view>("backend_config_str"));
  // VLOG(0) << backend_config_str;

  std::regex regexPattern(R"(")" + config_name + R"(":\{(.*?)\}\})");
  std::smatch matches;

  if (std::regex_search(backend_config_str, matches, regexPattern)) {
    if (matches.size() > 1) {
      std::string configContent = matches[1].str();
      // VLOG(0) << configContent;
      std::regex kvPattern("\"([^\"]+)\":([^[{,}]+)");
      std::sregex_iterator iter(configContent.begin(), configContent.end(),
                                kvPattern);
      std::sregex_iterator end;

      while (iter != end) {
        std::string key = (*iter)[1].str();
        std::string value = (*iter)[2].str();
        key.erase(std::remove(key.begin(), key.end(), '\"'), key.end());
        value.erase(std::remove(value.begin(), value.end(), '\"'), value.end());
        backend_dict[key] = value;
        // VLOG(0) << key << ": " << value;
        ++iter;
      }

      std::regex kvPattern_("\"([^\"]+)\":\\[(.*?)\\]");
      std::sregex_iterator iter_(configContent.begin(), configContent.end(),
                                 kvPattern_);
      std::sregex_iterator end_;

      while (iter_ != end_) {
        std::string key = (*iter_)[1].str();
        std::string value = (*iter_)[2].str();
        key.erase(std::remove(key.begin(), key.end(), '\"'), key.end());
        value.erase(std::remove(value.begin(), value.end(), '\"'), value.end());
        backend_dict[key] = value;
        // VLOG(0) << key << ": " << value;
        ++iter_;
      }
    }
  } else {
    return absl::InternalError(config_name + " not found.");
  }
  return absl::OkStatus();
}

absl::Status RunGpuConvCustomCall(
    se::Stream* stream, se::ScratchAllocator* scratch_allocator,
    std::vector<ffi::BufferBase>& operand_se_buffers,
    ffi::BufferBase& result_buffer, const ffi::Dictionary& dict,
    CudnnConvKind conv_kind) {
  std::string config_name("cudnn_conv_backend_config");
  absl::flat_hash_map<std::string, std::string> backend_dict;
  GetBackendDict(dict, backend_dict, config_name);
  TF_ASSIGN_OR_RETURN(auto conv_primitive,
                      GetOrCreateOneDnnConvPrimitive(
                          stream, dict, backend_dict, operand_se_buffers,
                          result_buffer, scratch_allocator, conv_kind));
  TF_RETURN_IF_ERROR(RunGpuConv(
      conv_primitive, dict, absl::MakeSpan(operand_se_buffers), result_buffer, conv_kind));
  return absl::OkStatus();
}

void GetDimensions(std::string dimensions_str, std::vector<int64_t>& dimensions){
  std::regex NumPattern("(\\d+)");
  std::sregex_iterator iter(dimensions_str.begin(), dimensions_str.end(),
                            NumPattern);
  std::sregex_iterator end;
  // VLOG(0) << "dimensions_str" << ": " << dimensions_str;
  while (iter != end) {
    std::string dimension = (*iter)[1].str();
    // VLOG(0) << "dimension" << ": " << dimension;
    dimensions.push_back(std::stoll(dimension));
    ++iter;
  }
}

absl::Status RunGemmCustomCall(ffi::BufferBase* lhs,
                               ffi::BufferBase* rhs,
                               ffi::BufferBase* add,
                               ffi::BufferBase* output,
                               ffi::BufferBase* bias,
                               se::Stream* stream, const ffi::Dictionary& dict,
                               absl::flat_hash_map<std::string, std::string>& backend_dict,
                               se::ScratchAllocator* scratch_allocator) {
  SYCLGemm::GemmBackendEpilogue epilogue;
  if (backend_dict.find("epilogue") == backend_dict.end()) {
    epilogue = SYCLGemm::GemmBackendEpilogue::DEFAULT;
  } else {
    TF_ASSIGN_OR_RETURN(epilogue,
                        SYCLGemm::EpilogueCast(backend_dict["epilogue"]));
  }

  se::DeviceMemoryBase lhs_data = lhs->data;
  se::DeviceMemoryBase rhs_data = rhs->data;
  se::DeviceMemoryBase output_data = output->data;
  se::DeviceMemoryBase add_data;
  se::DeviceMemoryBase bias_data;
  if(add != nullptr) add_data = add->data;
  if(bias != nullptr) bias_data = bias->data;

  Shape lhs_shape = ShapeUtil::MakeShape(lhs->dtype, lhs->dimensions);
  Shape rhs_shape = ShapeUtil::MakeShape(rhs->dtype, rhs->dimensions);
  Shape output_shape = ShapeUtil::MakeShape(output->dtype, output->dimensions);
  lhs_shape.mutable_layout()->clear_minor_to_major();
  for (int i = 0; i < lhs->dimensions.size(); ++i) {
    lhs_shape.mutable_layout()->add_minor_to_major(
      *dict.get<int64_t>("lhs_minor_to_major_" + std::to_string(i)));
  }
  rhs_shape.mutable_layout()->clear_minor_to_major();
  for (int i = 0; i < rhs->dimensions.size(); ++i) {
    rhs_shape.mutable_layout()->add_minor_to_major(
      *dict.get<int64_t>("rhs_minor_to_major_" + std::to_string(i)));
  }
  output_shape.mutable_layout()->clear_minor_to_major();
  for (int i = 0; i < output->dimensions.size(); ++i) {
    output_shape.mutable_layout()->add_minor_to_major(
      *dict.get<int64_t>("output_minor_to_major_" + std::to_string(i)));
  }

  std::vector<int64_t> lhs_batch_dims;
  std::vector<int64_t> lhs_contracting_dims;
  std::vector<int64_t> rhs_batch_dims;
  std::vector<int64_t> rhs_contracting_dims;
  GetDimensions(backend_dict["lhs_batch_dimensions"], lhs_batch_dims);
  GetDimensions(backend_dict["lhs_contracting_dimensions"], lhs_contracting_dims);
  GetDimensions(backend_dict["rhs_batch_dimensions"], rhs_batch_dims);
  GetDimensions(backend_dict["rhs_contracting_dimensions"], rhs_contracting_dims);

  double alpha_real = std::stod(backend_dict["alpha_real"]);
  double alpha_imag = std::stod(backend_dict["alpha_imag"]);
  double beta = std::stod(backend_dict["beta"]);
  int64_t algorithm = std::stol(backend_dict["selected_algorithm"]);
  int64_t compute_precision = 0; // not be used in sycl
  bool grad_x = (backend_dict["grad_x"] == "true");
  bool grad_y = (backend_dict["grad_y"] == "true");
  TF_ASSIGN_OR_RETURN(
      auto gemm_config,
      GemmConfig::For(lhs_shape, lhs_batch_dims, lhs_contracting_dims,
                      rhs_shape, rhs_batch_dims, rhs_contracting_dims,
                      output_shape,
                      alpha_real, alpha_imag, beta, algorithm, compute_precision,
                      grad_x, grad_y));
  return RunGemm(gemm_config, lhs_data, rhs_data, add_data, output_data, bias_data,
                 stream, epilogue, scratch_allocator);
}

}  // namespace gpu
}  // namespace xla