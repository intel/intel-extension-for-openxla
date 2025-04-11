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

#include "xla/service/gpu/onednn_gpu_conv_runner.h"

#include <string>

#include "xla/service/gpu/scratch_allocator.h"
#include "xla/service/gpu/stream_executor_util.h"

namespace xla {
namespace gpu {
using se::DeviceMemory;
using se::DeviceMemoryBase;
using se::Stream;
using se::dnn::AlgorithmConfig;
using se::dnn::BatchDescriptor;
using se::dnn::ConvolutionDescriptor;
using se::dnn::DataLayout;
using se::dnn::DimIndex;
using se::dnn::FilterDescriptor;
using se::dnn::FilterLayout;
using se::dnn::ProfileResult;

using ConvFwdPd = dnnl::convolution_forward::primitive_desc;
using ConvBwdInputPd = dnnl::convolution_backward_data::primitive_desc;
using ConvBwdFilterPd = dnnl::convolution_backward_weights::primitive_desc;
using ConvBwdFilterPrimitive = dnnl::convolution_backward_weights;

namespace {

int64_t GetVectCSize(DataLayout layout) {
  switch (layout) {
    case DataLayout::kBatchDepthYX4:
      return 4;
    case DataLayout::kBatchDepthYX32:
      return 32;
    default:
      return 1;
  }
}

int64_t GetVectCSize(FilterLayout layout) {
  switch (layout) {
    case FilterLayout::kOutputInputYX4:
      return 4;
    case FilterLayout::kOutputInputYX32:
      return 32;
    default:
      return 1;
  }
}

absl::Status CreateOneDnnPrimitive(
    OneDnnConvPrimitive* onednn_primitive,  // NOLINT
    const ffi::Dictionary& dict,
    absl::Span<const ffi::BufferBase> operand_buffers,
    ffi::BufferBase result_buffer, se::Stream* stream,
    se::ScratchAllocator* scratch_allocator, CudnnConvKind conv_kind) {
  sycl::queue* dpcpp_stream = se::gpu::AsGpuStreamValue(stream);
  onednn_primitive->engine = FindOrCreateEngine(dpcpp_stream);
  onednn_primitive->stream =
      dnnl::sycl_interop::make_stream(onednn_primitive->engine, *dpcpp_stream);
  DataLayout input_dl = static_cast<DataLayout>(*dict.get<int32_t>("input_dl"));
  FilterLayout filter_dl =
      static_cast<FilterLayout>(*dict.get<int32_t>("filter_dl"));
  DataLayout output_dl =
      static_cast<DataLayout>(*dict.get<int32_t>("output_dl"));

  PrimitiveType input_type, filter_type, output_type;
  absl::Span<const int64_t> input_dimensions, filter_dimensions,
      output_dimensions;
  void* input_data;
  void* filter_data;
  void* output_data;
  void* bias_data = nullptr;
  void* side_input_data = nullptr;

  float conv_result_scale = *dict.get<float>("conv_result_scale");
  bool conv_result_scale_one = (fabs(conv_result_scale - 1.0f) < 1e-6);

  switch (conv_kind) {
    case CudnnConvKind::kForward:
    case CudnnConvKind::kForwardActivation:
      input_type = operand_buffers[0].dtype;

      input_dimensions = operand_buffers[0].dimensions;
      filter_dimensions = operand_buffers[1].dimensions;
      output_dimensions = result_buffer.dimensions;

      input_data = const_cast<void*>(operand_buffers[0].data.opaque());
      filter_data = const_cast<void*>(operand_buffers[1].data.opaque());
      output_data = const_cast<void*>(result_buffer.data.opaque());
      break;
    case CudnnConvKind::kBackwardInput:
      input_type = result_buffer.dtype;

      input_dimensions = result_buffer.dimensions;
      filter_dimensions = operand_buffers[1].dimensions;
      output_dimensions = operand_buffers[0].dimensions;

      input_data = const_cast<void*>(result_buffer.data.opaque());
      filter_data = const_cast<void*>(operand_buffers[1].data.opaque());
      output_data = const_cast<void*>(operand_buffers[0].data.opaque());
      break;
    case CudnnConvKind::kBackwardFilter:
      input_type = operand_buffers[0].dtype;

      input_dimensions = operand_buffers[0].dimensions;
      filter_dimensions = result_buffer.dimensions;
      output_dimensions = operand_buffers[1].dimensions;

      input_data = const_cast<void*>(operand_buffers[0].data.opaque());
      filter_data = const_cast<void*>(result_buffer.data.opaque());
      output_data = const_cast<void*>(operand_buffers[1].data.opaque());
      break;
    default:
      return Internal("Unkown convolution kind");
  }

  float side_input_scale;
  bool side_input_scale_zero;
  if (conv_kind == CudnnConvKind::kForwardActivation) {
    bias_data = const_cast<void*>(operand_buffers[2].data.opaque());
    if (operand_buffers.size() >= 4) {
      side_input_data = const_cast<void*>(operand_buffers[3].data.opaque());
      side_input_scale = *dict.get<float>("side_input_scale");
      side_input_scale_zero = (fabs(side_input_scale - 0.0f) < 1e-6);
    }
  }

  const int num_dimensions = *dict.get<int32_t>("window_num_dimensions");
  CHECK_LE(num_dimensions, 3);

  // OneDNN does not support 1D convolutions. We therefore express 1D
  // convolutions as 2D convolutions where the first spatial dimension is 1.
  // This matches the behavior of TF (see definition of conv1d in
  // tensorflow/python/ops/nn_ops.py).
  const int effective_num_dimensions = std::max(2, num_dimensions);

  int ic = GetVectCSize(input_dl) *
           input_dimensions[*dict.get<int64_t>("input_feature_dimension")];
  int n = input_dimensions[*dict.get<int64_t>("input_batch_dimension")];
  int id, ih, iw;
  if (num_dimensions == 3) {
    id = input_dimensions[*dict.get<int64_t>("input_spatial_dimensions_0")];
    ih = input_dimensions[*dict.get<int64_t>("input_spatial_dimensions_1")];
    iw = input_dimensions[*dict.get<int64_t>("input_spatial_dimensions_2")];
  } else if (num_dimensions == 2) {
    ih = input_dimensions[*dict.get<int64_t>("input_spatial_dimensions_0")];
    iw = input_dimensions[*dict.get<int64_t>("input_spatial_dimensions_1")];
  } else if (num_dimensions == 1) {
    ih = 1;
    iw = input_dimensions[*dict.get<int64_t>("input_spatial_dimensions_0")];
  } else if (num_dimensions == 0) {
    ih = 1;
    iw = 1;
  } else {
    return Internal("Invalid convolution dimension num");
  }

  int kd, kh, kw;
  if (num_dimensions == 3) {
    kd = filter_dimensions[*dict.get<int64_t>("kernel_spatial_dimensions_0")];
    kh = filter_dimensions[*dict.get<int64_t>("kernel_spatial_dimensions_1")];
    kw = filter_dimensions[*dict.get<int64_t>("kernel_spatial_dimensions_2")];
  } else if (num_dimensions == 2) {
    kh = filter_dimensions[*dict.get<int64_t>("kernel_spatial_dimensions_0")];
    kw = filter_dimensions[*dict.get<int64_t>("kernel_spatial_dimensions_1")];
  } else if (num_dimensions == 1) {
    kh = 1;
    kw = filter_dimensions[*dict.get<int64_t>("kernel_spatial_dimensions_0")];
  } else if (num_dimensions == 0) {
    kh = 1;
    kw = 1;
  } else {
    return Internal("Invalid convolution dimension num");
  }

  // It is group-conv if filter_in != src_in
  // G = src_in/filter_in
  // O = filter_out/G
  // TODO: depthwise-conv
  int filter_ic =
      filter_dimensions[*dict.get<int64_t>("kernel_input_feature_dimension")];
  int filter_oc =
      filter_dimensions[*dict.get<int64_t>("kernel_output_feature_dimension")];
  bool is_group_conv = ic != filter_ic;
  int kg = ic / filter_ic;  // kg for group-conv and depthwise-conv
  int ko = filter_oc / kg;
  int ki = filter_ic;

  int padding_d_l, padding_h_l, padding_w_l;
  int padding_d_h, padding_h_h, padding_w_h;
  int stride_d, stride_h, stride_w, dilate_d, dilate_h, dilate_w;

  if (num_dimensions == 3) {
    padding_d_l = *dict.get<int64_t>("window_padding_low_0");
    padding_h_l = *dict.get<int64_t>("window_padding_low_1");
    padding_w_l = *dict.get<int64_t>("window_padding_low_2");
    padding_d_h = *dict.get<int64_t>("window_padding_high_0");
    padding_h_h = *dict.get<int64_t>("window_padding_high_1");
    padding_w_h = *dict.get<int64_t>("window_padding_high_2");

    stride_d = *dict.get<int64_t>("window_stride_0");
    stride_h = *dict.get<int64_t>("window_stride_1");
    stride_w = *dict.get<int64_t>("window_stride_2");

    dilate_d = *dict.get<int64_t>("window_dilation_0");
    dilate_h = *dict.get<int64_t>("window_dilation_1");
    dilate_w = *dict.get<int64_t>("window_dilation_2");
  } else if (num_dimensions == 2) {
    padding_h_l = *dict.get<int64_t>("window_padding_low_0");
    padding_w_l = *dict.get<int64_t>("window_padding_low_1");
    padding_h_h = *dict.get<int64_t>("window_padding_high_0");
    padding_w_h = *dict.get<int64_t>("window_padding_high_1");

    stride_h = *dict.get<int64_t>("window_stride_0");
    stride_w = *dict.get<int64_t>("window_stride_1");

    dilate_h = *dict.get<int64_t>("window_dilation_0");
    dilate_w = *dict.get<int64_t>("window_dilation_1");
  } else if (num_dimensions == 1) {
    padding_h_l = 0;
    padding_w_l = *dict.get<int64_t>("window_padding_low_0");
    padding_h_h = 0;
    padding_w_h = *dict.get<int64_t>("window_padding_high_0");

    stride_h = 1;
    stride_w = *dict.get<int64_t>("window_stride_0");

    dilate_h = 1;
    dilate_w = *dict.get<int64_t>("window_dilation_0");
  } else if (num_dimensions == 0) {
    padding_h_l = 0;
    padding_w_l = 0;
    padding_h_h = 0;
    padding_w_h = 0;

    stride_h = 1;
    stride_w = 1;

    dilate_h = 1;
    dilate_w = 1;
  }

  int od, oh, ow;
  int oc = output_dimensions[*dict.get<int64_t>("output_feature_dimension")];
  if (num_dimensions == 3) {
    od = output_dimensions[*dict.get<int64_t>("output_spatial_dimensions_0")];
    oh = output_dimensions[*dict.get<int64_t>("output_spatial_dimensions_1")];
    ow = output_dimensions[*dict.get<int64_t>("output_spatial_dimensions_2")];
  } else if (num_dimensions == 2) {
    oh = output_dimensions[*dict.get<int64_t>("output_spatial_dimensions_0")];
    ow = output_dimensions[*dict.get<int64_t>("output_spatial_dimensions_1")];
  } else if (num_dimensions == 1) {
    oh = 1;
    ow = output_dimensions[*dict.get<int64_t>("output_spatial_dimensions_0")];
  } else if (num_dimensions == 0) {
    oh = 1;
    ow = 1;
  }
  bool is_conv3d = (num_dimensions == 3);
  try {
    dnnl::memory::dims src_dims, filter_dims, bias_dims, dst_dims, stride_dims,
        padding_dims_l, padding_dims_r, dilation_dims;
    dnnl::memory::format_tag src_fmt, weight_fmt, dst_fmt;
    if (!is_conv3d) {
      src_dims = {n, ic, ih, iw};
      if (is_group_conv)
        filter_dims = {kg, ko, ki, kh, kw};
      else
        filter_dims = {ko, ki, kh, kw};
      bias_dims = {oc};
      dst_dims = {n, oc, oh, ow};
      stride_dims = {stride_h, stride_w};
      padding_dims_l = {padding_h_l, padding_w_l};
      padding_dims_r = {padding_h_h, padding_w_h};
      dilation_dims = {dilate_h - 1, dilate_w - 1};

      switch (input_dl) {
        case DataLayout::kBatchDepthYX:
          src_fmt = dnnl::memory::format_tag::nchw;
          break;
        case DataLayout::kBatchYXDepth:
          src_fmt = dnnl::memory::format_tag::nhwc;
          break;
        default:
          return Internal("Unsupported convolution input format");
      }

      switch (filter_dl) {
        case FilterLayout::kOutputInputYX:
          weight_fmt = is_group_conv ? dnnl::memory::format_tag::goihw
                                     : dnnl::memory::format_tag::oihw;
          break;
        case FilterLayout::kOutputYXInput:
          weight_fmt = is_group_conv ? dnnl::memory::format_tag::gohwi
                                     : dnnl::memory::format_tag::ohwi;
          break;
        case FilterLayout::kYXInputOutput:
          weight_fmt = is_group_conv ? dnnl::memory::format_tag::hwigo
                                     : dnnl::memory::format_tag::hwio;
          break;
        default:
          return Internal("Unsupported convolution weight format");
      }

      switch (output_dl) {
        case DataLayout::kBatchDepthYX:
          dst_fmt = dnnl::memory::format_tag::nchw;
          break;
        case DataLayout::kBatchYXDepth:
          dst_fmt = dnnl::memory::format_tag::nhwc;
          break;
        default:
          return Internal("Unsupported convolution output format");
      }
    } else {
      src_dims = {n, ic, id, ih, iw};
      if (is_group_conv)
        filter_dims = {kg, ko, ki, kd, kh, kw};
      else
        filter_dims = {ko, ki, kd, kh, kw};
      bias_dims = {oc};
      dst_dims = {n, oc, od, oh, ow};
      stride_dims = {stride_d, stride_h, stride_w};
      padding_dims_l = {padding_d_l, padding_h_l, padding_w_l};
      padding_dims_r = {padding_d_h, padding_h_h, padding_w_h};
      dilation_dims = {dilate_d - 1, dilate_h - 1, dilate_w - 1};

      switch (input_dl) {
        case DataLayout::kBatchDepthYX:
          src_fmt = dnnl::memory::format_tag::ncdhw;
          break;
        case DataLayout::kBatchYXDepth:
          src_fmt = dnnl::memory::format_tag::ndhwc;
          break;
        default:
          return Internal("Unsupported convolution input format");
      }

      switch (filter_dl) {
        case FilterLayout::kOutputInputYX:
          weight_fmt = is_group_conv ? dnnl::memory::format_tag::goidhw
                                     : dnnl::memory::format_tag::oidhw;
          break;
        case FilterLayout::kOutputYXInput:
          weight_fmt = is_group_conv ? dnnl::memory::format_tag::godhwi
                                     : dnnl::memory::format_tag::odhwi;
          break;
        default:
          return Internal("Unsupported convolution weight format");
      }

      switch (output_dl) {
        case DataLayout::kBatchDepthYX:
          dst_fmt = dnnl::memory::format_tag::ncdhw;
          break;
        case DataLayout::kBatchYXDepth:
          dst_fmt = dnnl::memory::format_tag::ndhwc;
          break;
        default:
          return Internal("Unsupported convolution output format");
      }
    }

    auto kind = dnnl::sycl_interop::memory_kind::usm;

    dnnl::memory::data_type data_type;

    switch (input_type) {
      case BF16:
        data_type = dnnl::memory::data_type::bf16;
        break;
      case F32:
        data_type = dnnl::memory::data_type::f32;
        break;
      case F16:
        data_type = dnnl::memory::data_type::f16;
        break;
      case F64:
        data_type = dnnl::memory::data_type::f64;
        break;
      case S8:
        data_type = dnnl::memory::data_type::s8;
        break;
      case S32:
        data_type = dnnl::memory::data_type::s32;
        break;
      default:
        return Internal("Unsupported convolution input data type");
    }

    dnnl::memory::desc src_md =
        dnnl::memory::desc({src_dims}, data_type, src_fmt);
    dnnl::memory::desc filter_md =
        dnnl::memory::desc({filter_dims}, data_type, weight_fmt);
    dnnl::memory::desc dst_md =
        dnnl::memory::desc({dst_dims}, data_type, dst_fmt);

    bool flag = false;
    tsl::ReadBoolFromEnvVar("ONEDNN_PLAIN_WEIGHT", false, &flag);
    dnnl::memory::desc filter_md_prefer = dnnl::memory::desc(
        {filter_dims}, data_type, dnnl::memory::format_tag::any);
    if (flag)
      filter_md_prefer =
          dnnl::memory::desc({filter_dims}, data_type, weight_fmt);

    onednn_primitive->src_memory = dnnl::sycl_interop::make_memory(
        src_md, onednn_primitive->engine, kind, input_data);
    onednn_primitive->filter_memory = dnnl::sycl_interop::make_memory(
        filter_md, onednn_primitive->engine, kind, filter_data);
    onednn_primitive->dst_memory = dnnl::sycl_interop::make_memory(
        dst_md, onednn_primitive->engine, kind, output_data);

    // if alpha is 1:
    //   out = activation(conv(x, w, bias) + beta * side)
    //   po.append_sum(beta)
    //   po.append_eltwise(dnnl::algorithm::activation, 1, 0);
    // else:
    //   out = activation(alpha * conv(x, w) + beta * side + bias)
    //   po.append_eltwise(dnnl::algorithm::eltwise_linear, alpha, 0);
    //   po.append_sum(beta)
    //   po.append_binary(1, bias);
    //   po.append_eltwise(dnnl::algorithm::activation, 1, 0);
    dnnl::post_ops po;
    dnnl::primitive_attr post_ops_attr;
    if (!conv_result_scale_one)
      po.append_eltwise(dnnl::algorithm::eltwise_linear, conv_result_scale, 0);
    if (side_input_data && !side_input_scale_zero)
      po.append_sum(side_input_scale);
    if (!conv_result_scale_one && bias_data) {
      auto bias_post_md =
          dnnl::memory::desc(bias_dims, data_type, dnnl::memory::format_tag::x);
      po.append_binary(dnnl::algorithm::binary_add, bias_post_md);
      onednn_primitive->bias_memory = dnnl::sycl_interop::make_memory(
          bias_post_md, onednn_primitive->engine, kind, bias_data);
      onednn_primitive->fwd_primitives_args.insert(
          {DNNL_ARG_ATTR_MULTIPLE_POST_OP(po.len() - 1) | DNNL_ARG_SRC_1,
           onednn_primitive->bias_memory});
    }
    if (conv_kind == CudnnConvKind::kForwardActivation) {
      auto activation_mode = static_cast<stream_executor::dnn::ActivationMode>(
          *dict.get<int32_t>("activation_mode"));
      switch (activation_mode) {
        case stream_executor::dnn::kSigmoid:
          po.append_eltwise(dnnl::algorithm::eltwise_logistic, 1, 0);
          break;
        case stream_executor::dnn::kRelu:
          po.append_eltwise(dnnl::algorithm::eltwise_relu, 0, 0);
          break;
        case stream_executor::dnn::kRelu6:
          po.append_eltwise(dnnl::algorithm::eltwise_clip_v2, 0, 6);
          break;
        case stream_executor::dnn::kTanh:
          po.append_eltwise(dnnl::algorithm::eltwise_tanh, 0, 0);
          break;
        case stream_executor::dnn::kElu:
          po.append_eltwise(dnnl::algorithm::eltwise_elu, 1, 0);
          break;
        case stream_executor::dnn::kLeakyRelu:
          po.append_eltwise(dnnl::algorithm::eltwise_relu,
                            *dict.get<float>("leakyrelu_alpha"), 0);
          break;
        case stream_executor::dnn::kNone:
          break;
        default:
          return Internal("Unsupported Activation mode");
      }
    }
    post_ops_attr.set_post_ops(po);
    post_ops_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    // Set fp32 mode.
    dnnl::fpmath_mode fp32_math_mode = GetFP32MathMode();
    if (input_type == F32) {
      post_ops_attr.set_fpmath_mode(fp32_math_mode);
    }

    if (conv_kind == CudnnConvKind::kForward ||
        conv_kind == CudnnConvKind::kForwardActivation) {
      ConvFwdPd fwd_pd;
      if (bias_data != nullptr && conv_result_scale_one) {
        auto bias_md = dnnl::memory::desc(bias_dims, data_type,
                                          dnnl::memory::format_tag::x);
        fwd_pd = ConvFwdPd(onednn_primitive->engine, dnnl::prop_kind::forward,
                           dnnl::algorithm::convolution_direct, src_md,
                           filter_md_prefer, bias_md, dst_md, stride_dims,
                           dilation_dims, padding_dims_l, padding_dims_r,
                           post_ops_attr);
        onednn_primitive->bias_memory = dnnl::sycl_interop::make_memory(
            bias_md, onednn_primitive->engine, kind, bias_data);
        onednn_primitive->fwd_primitives_args.insert(
            {DNNL_ARG_BIAS, onednn_primitive->bias_memory});
      } else {
        fwd_pd = ConvFwdPd(onednn_primitive->engine, dnnl::prop_kind::forward,
                           dnnl::algorithm::convolution_direct, src_md,
                           filter_md_prefer, dst_md, stride_dims, dilation_dims,
                           padding_dims_l, padding_dims_r, post_ops_attr);
      }

      onednn_primitive->fwd_primitive = dnnl::convolution_forward(fwd_pd);
      size_t scratchpad_size = fwd_pd.scratchpad_desc().get_size();
      void* workspace;
      TF_RETURN_IF_ERROR(
          AllocateWorkspace(&workspace, scratch_allocator, scratchpad_size));
      onednn_primitive->scratchpad_memory = dnnl::memory(
          fwd_pd.scratchpad_desc(), onednn_primitive->engine, workspace);

      bool is_filter_reordered = (filter_md != fwd_pd.weights_desc());
      if (is_filter_reordered) {
        onednn_primitive->has_reorder = true;
        size_t reorder_filter_data_size = fwd_pd.weights_desc().get_size();
        void* reorder_filter;
        TF_RETURN_IF_ERROR(AllocateWorkspace(&reorder_filter, scratch_allocator,
                                             reorder_filter_data_size));

        onednn_primitive->internal_filter_memory = dnnl::memory(
            fwd_pd.weights_desc(), onednn_primitive->engine, reorder_filter);
        onednn_primitive->filter_reorder_primitive =
            dnnl::reorder(onednn_primitive->filter_memory,
                          onednn_primitive->internal_filter_memory);
        onednn_primitive->reorder_args = {
            {DNNL_ARG_SRC, onednn_primitive->filter_memory},
            {DNNL_ARG_DST, onednn_primitive->internal_filter_memory}};

        onednn_primitive->fwd_primitives_args.insert(
            {DNNL_ARG_WEIGHTS, onednn_primitive->internal_filter_memory});
      } else {
        onednn_primitive->has_reorder = false;
        onednn_primitive->fwd_primitives_args.insert(
            {DNNL_ARG_WEIGHTS, onednn_primitive->filter_memory});
      }
      onednn_primitive->fwd_primitives_args.insert(
          {DNNL_ARG_SRC, onednn_primitive->src_memory});
      onednn_primitive->fwd_primitives_args.insert(
          {DNNL_ARG_DST, onednn_primitive->dst_memory});
      onednn_primitive->fwd_primitives_args.insert(
          {DNNL_ARG_SCRATCHPAD, onednn_primitive->scratchpad_memory});

    } else if (conv_kind == CudnnConvKind::kBackwardInput) {
      // TODO: handle post_ops_attr.
      ConvFwdPd fwd_pd = ConvFwdPd(
          onednn_primitive->engine, dnnl::prop_kind::forward,
          dnnl::algorithm::convolution_direct, src_md, filter_md_prefer, dst_md,
          stride_dims, dilation_dims, padding_dims_l, padding_dims_r);

      dnnl::primitive_attr attr;
      attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
      ConvBwdInputPd bwd_input_pd = ConvBwdInputPd(
          onednn_primitive->engine, dnnl::algorithm::convolution_direct, src_md,
          filter_md_prefer, dst_md, stride_dims, dilation_dims, padding_dims_l,
          padding_dims_r, fwd_pd, attr);

      size_t scratchpad_size = bwd_input_pd.scratchpad_desc().get_size();
      void* workspace;
      TF_RETURN_IF_ERROR(
          AllocateWorkspace(&workspace, scratch_allocator, scratchpad_size));
      onednn_primitive->scratchpad_memory = dnnl::memory(
          bwd_input_pd.scratchpad_desc(), onednn_primitive->engine, workspace);

      bool is_filter_reordered = (filter_md != bwd_input_pd.weights_desc());
      if (is_filter_reordered) {
        size_t reorder_filter_data_size =
            bwd_input_pd.weights_desc().get_size();
        void* reorder_filter;
        TF_RETURN_IF_ERROR(AllocateWorkspace(&reorder_filter, scratch_allocator,
                                             reorder_filter_data_size));

        onednn_primitive->internal_filter_memory =
            dnnl::memory(bwd_input_pd.weights_desc(), onednn_primitive->engine,
                         reorder_filter);
        onednn_primitive->filter_reorder_primitive =
            dnnl::reorder(onednn_primitive->filter_memory,
                          onednn_primitive->internal_filter_memory);
        onednn_primitive->reorder_args = {
            {DNNL_ARG_SRC, onednn_primitive->filter_memory},
            {DNNL_ARG_DST, onednn_primitive->internal_filter_memory}};
        onednn_primitive->bwd_input_primitive_args.insert(
            {DNNL_ARG_WEIGHTS, onednn_primitive->internal_filter_memory});
        onednn_primitive->has_reorder = true;
      } else {
        onednn_primitive->bwd_input_primitive_args.insert(
            {DNNL_ARG_WEIGHTS, onednn_primitive->filter_memory});
        onednn_primitive->has_reorder = false;
      }

      onednn_primitive->bwd_input_primitive_args.insert(
          {DNNL_ARG_DIFF_DST, onednn_primitive->dst_memory});
      onednn_primitive->bwd_input_primitive_args.insert(
          {DNNL_ARG_DIFF_SRC, onednn_primitive->src_memory});
      onednn_primitive->bwd_input_primitive_args.insert(
          {DNNL_ARG_SCRATCHPAD, onednn_primitive->scratchpad_memory});

      onednn_primitive->bwd_input_primitive =
          dnnl::convolution_backward_data(bwd_input_pd);

    } else if (conv_kind == CudnnConvKind::kBackwardFilter) {
      // TODO: handle post_ops_attr.
      ConvFwdPd fwd_pd = ConvFwdPd(
          onednn_primitive->engine, dnnl::prop_kind::forward,
          dnnl::algorithm::convolution_direct, src_md, filter_md_prefer, dst_md,
          stride_dims, dilation_dims, padding_dims_l, padding_dims_r);

      dnnl::primitive_attr attr;
      attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
      ConvBwdFilterPd bwd_filter_pd = ConvBwdFilterPd(
          onednn_primitive->engine, dnnl::algorithm::convolution_direct, src_md,
          filter_md_prefer, dst_md, stride_dims, dilation_dims, padding_dims_l,
          padding_dims_r, fwd_pd, attr);

      size_t scratchpad_size = bwd_filter_pd.scratchpad_desc().get_size();
      void* workspace;
      TF_RETURN_IF_ERROR(
          AllocateWorkspace(&workspace, scratch_allocator, scratchpad_size));
      onednn_primitive->scratchpad_memory = dnnl::memory(
          bwd_filter_pd.scratchpad_desc(), onednn_primitive->engine, workspace);

      bool is_filter_reordered =
          (filter_md != bwd_filter_pd.diff_weights_desc());
      if (is_filter_reordered) {
        onednn_primitive->has_reorder = true;
        size_t reorder_filter_data_size =
            bwd_filter_pd.diff_weights_desc().get_size();
        void* prefer_filter;
        TF_RETURN_IF_ERROR(AllocateWorkspace(&prefer_filter, scratch_allocator,
                                             reorder_filter_data_size));

        onednn_primitive->internal_filter_memory =
            dnnl::memory(bwd_filter_pd.diff_weights_desc(),
                         onednn_primitive->engine, prefer_filter);
        onednn_primitive->filter_reorder_primitive =
            dnnl::reorder(onednn_primitive->internal_filter_memory,
                          onednn_primitive->filter_memory);
        onednn_primitive->reorder_args = {
            {DNNL_ARG_SRC, onednn_primitive->internal_filter_memory},
            {DNNL_ARG_DST, onednn_primitive->filter_memory}};

        onednn_primitive->bwd_filter_primitive_args.insert(
            {DNNL_ARG_DIFF_WEIGHTS, onednn_primitive->internal_filter_memory});
      } else {
        onednn_primitive->has_reorder = false;
        onednn_primitive->bwd_filter_primitive_args.insert(
            {DNNL_ARG_DIFF_WEIGHTS, onednn_primitive->filter_memory});
      }

      onednn_primitive->bwd_filter_primitive_args.insert(
          {DNNL_ARG_SRC, onednn_primitive->src_memory});
      onednn_primitive->bwd_filter_primitive_args.insert(
          {DNNL_ARG_DIFF_DST, onednn_primitive->dst_memory});
      onednn_primitive->bwd_filter_primitive_args.insert(
          {DNNL_ARG_SCRATCHPAD, onednn_primitive->scratchpad_memory});

      onednn_primitive->bwd_filter_primitive =
          ConvBwdFilterPrimitive(bwd_filter_pd);

    } else {
      return Internal("Unkown convolutuion kind");
    }
  } catch (dnnl::error& e) {
    return Internal("OneDNN Conv error: %s", e.message);
  }
  return absl::OkStatus();
}  // NOLINT
}  // namespace

absl::StatusOr<OneDnnConvPrimitive> GetOrCreateOneDnnConvPrimitive(
    se::Stream* stream, const ffi::Dictionary& dict,
    const std::vector<ffi::BufferBase>& operand_se_buffers,
    const ffi::BufferBase& result_buffer,
    se::ScratchAllocator* scratch_allocator, CudnnConvKind conv_kind) {
  OneDnnConvPrimitive primitive;
  auto status = CreateOneDnnPrimitive(
      &primitive, dict, absl::MakeSpan(operand_se_buffers), result_buffer,
      stream, scratch_allocator, conv_kind);
  if (TF_PREDICT_FALSE(!status.ok())) {
    return status;
  }
  return primitive;
}

absl::Status RunGpuConv(const OneDnnConvPrimitive& onednn_primitive,
                        const ffi::Dictionary& dict,
                        absl::Span<const ffi::BufferBase> operand_buffers,
                        ffi::BufferBase result_buffer,
                        CudnnConvKind conv_kind) {
  void* input_data;
  void* filter_data;
  void* output_data;
  void* bias_data = nullptr;
  void* side_input_data = nullptr;

  switch (conv_kind) {
    case CudnnConvKind::kForward:
    case CudnnConvKind::kForwardActivation:
      input_data = const_cast<void*>(operand_buffers[0].data.opaque());
      filter_data = const_cast<void*>(operand_buffers[1].data.opaque());
      output_data = const_cast<void*>(result_buffer.data.opaque());
      break;
    case CudnnConvKind::kBackwardInput:
      input_data = const_cast<void*>(result_buffer.data.opaque());
      filter_data = const_cast<void*>(operand_buffers[1].data.opaque());
      output_data = const_cast<void*>(operand_buffers[0].data.opaque());

      break;
    case CudnnConvKind::kBackwardFilter:
      input_data = const_cast<void*>(operand_buffers[0].data.opaque());
      filter_data = const_cast<void*>(result_buffer.data.opaque());
      output_data = const_cast<void*>(operand_buffers[1].data.opaque());
      break;
    default:
      return Internal("Unkown convolution kind");
  }

  if (conv_kind == CudnnConvKind::kForwardActivation) {
    bias_data = const_cast<void*>(operand_buffers[2].data.opaque());
    if (operand_buffers.size() >= 4) {
      side_input_data = const_cast<void*>(operand_buffers[3].data.opaque());
    }
  }
  onednn_primitive.src_memory.set_data_handle(input_data);
  onednn_primitive.filter_memory.set_data_handle(filter_data);
  onednn_primitive.dst_memory.set_data_handle(output_data);
  if (bias_data != nullptr) {
    onednn_primitive.bias_memory.set_data_handle(bias_data);
  }
  try {
    if (conv_kind == CudnnConvKind::kForward ||
        conv_kind == CudnnConvKind::kForwardActivation) {
      if (onednn_primitive.has_reorder) {
        onednn_primitive.filter_reorder_primitive.execute(
            onednn_primitive.stream, onednn_primitive.reorder_args);
      }
      onednn_primitive.fwd_primitive.execute(
          onednn_primitive.stream, onednn_primitive.fwd_primitives_args);
    } else if (conv_kind == CudnnConvKind::kBackwardInput) {
      if (onednn_primitive.has_reorder) {
        onednn_primitive.filter_reorder_primitive.execute(
            onednn_primitive.stream, onednn_primitive.reorder_args);
      }
      onednn_primitive.bwd_input_primitive.execute(
          onednn_primitive.stream, onednn_primitive.bwd_input_primitive_args);
    } else if (conv_kind == CudnnConvKind::kBackwardFilter) {
      onednn_primitive.bwd_filter_primitive.execute(
          onednn_primitive.stream, onednn_primitive.bwd_filter_primitive_args);
      if (onednn_primitive.has_reorder) {
        onednn_primitive.filter_reorder_primitive.execute(
            onednn_primitive.stream, onednn_primitive.reorder_args);
      }
    } else {
      return Internal("Unkown convolutuion kind");
    }
  } catch (dnnl::error& e) {
    std::string error_msg = "Status: " + std::to_string(e.status) +
                            ", message: " + std::string(e.message) +
                            ", in file " + std::string(__FILE__) + ":" +
                            std::to_string(__LINE__);
    std::cout << error_msg << std::endl;
  }

  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla