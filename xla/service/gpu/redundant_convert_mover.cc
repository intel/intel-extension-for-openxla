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

#include "xla/service/gpu/redundant_convert_mover.h"

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/pattern_matcher.h"

namespace xla {
namespace gpu {

namespace {
namespace m = match;

template <typename Pattern>
auto OptionalBitcast(Pattern pattern) {
  return m::AnyOf<HloInstruction>(m::Bitcast(pattern), std::move(pattern));
}

bool IsConvert(const HloInstruction* instr) {
  return instr->opcode() == HloOpcode::kConvert;
}

bool MatchDuplicateConvertPatterns(HloInstruction* instr,
                                   HloInstruction** bitcast_input,
                                   HloInstruction** convert_2) {
  // try to match convert(optionalbitcast(convert(optionalbitcast(input))))
  // where input's shape and element type is same as the final output
  auto default_duplicate_convert_pattern =
      m::Op(convert_2).WithPredicate(IsConvert).WithOneUse().WithOperand(
          0, OptionalBitcast(
                 m::Op()
                     .WithOperand(0, OptionalBitcast(m::Op(bitcast_input)))
                     .WithPredicate(IsConvert)
                     .WithOneUse()));
  if (Match(instr, default_duplicate_convert_pattern) &&
      (*convert_2)->shape() == (*bitcast_input)->shape() &&
      (*convert_2)->shape().element_type() ==
          (*bitcast_input)->shape().element_type()) {
    return true;
  }
  return false;
}

StatusOr<bool> RemoveRedundantConversion(HloInstruction* instr) {
  HloInstruction* bitcast_input = nullptr;
  HloInstruction* convert_2 = nullptr;
  if (MatchDuplicateConvertPatterns(instr, &bitcast_input, &convert_2)) {
    instr->ReplaceOperandWith(0, bitcast_input);
    return true;
  }
  return false;
}

}  // namespace

StatusOr<bool> RedundantConvertMover::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool any_changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instr : computation->MakeInstructionPostOrder()) {
      bool changed = false;
      TF_ASSIGN_OR_RETURN(changed, RemoveRedundantConversion(instr));
      any_changed |= changed;
    }
  }
  return any_changed;
}

}  // namespace gpu
}  // namespace xla