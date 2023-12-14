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

#include "xla/service/gpu/redundant_bitcast_mover.h"

#include "xla/hlo/ir/hlo_instruction.h"

namespace xla {
namespace gpu {

namespace {

StatusOr<bool> RunOnComputation(HloComputation& computation) {
  bool changed = false;
  for (HloInstruction* instruction : computation.MakeInstructionPostOrder()) {
    HloInstruction* input = instruction;
    while (input->opcode() == HloOpcode::kBitcast &&
           input->shape() == input->operand(0)->shape()) {
      input = input->mutable_operand(0);
    }

    if (input == instruction) continue;

    TF_RETURN_IF_ERROR(
        instruction->parent()->ReplaceInstruction(instruction, input));
    changed = true;
  }
  return changed;
}

}  // namespace

StatusOr<bool> RedundantBitcastMover::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeComputationPostOrder(execution_threads)) {
    TF_ASSIGN_OR_RETURN(bool comp_changed, RunOnComputation(*computation));
    changed |= comp_changed;
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla