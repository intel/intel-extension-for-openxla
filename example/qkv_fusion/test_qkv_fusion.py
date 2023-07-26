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

import jax
from jax import lax
from jax import random
import jax.numpy as jnp
import jax._src.test_util as jtu

import numpy as np

def seperateQKVGEMM(input, weight_q, weight_k, weight_v):
    out_q = jax.numpy.matmul(input, weight_q)
    out_k = jax.numpy.matmul(input, weight_k)
    out_v = jax.numpy.matmul(input, weight_v)
    return out_q, out_k, out_v

@jax.jit
def fusedQKVGEMM(input, weight_q, weight_k, weight_v):
    out_q = jax.numpy.matmul(input, weight_q)
    out_k = jax.numpy.matmul(input, weight_k)
    out_v = jax.numpy.matmul(input, weight_v)
    return out_q, out_k, out_v

def testQKVFusion():
    # Inputs
    m = 4
    k = 4096
    n = 4096
    key = jax.random.PRNGKey(1701)
    input = jax.random.uniform(key, (4, 4096)).astype(jnp.float16)
    weight_q = jax.random.uniform(key, (k, k)).astype(jnp.float16) # weights for Q
    weight_k = jax.random.uniform(key, (k, k)).astype(jnp.float16) # weights for K 
    weight_v = jax.random.uniform(key, (k, k)).astype(jnp.float16) # weights for V
    cpu_q, cpu_k, cpu_v = seperateQKVGEMM(input, weight_q, weight_k, weight_v)
    xpu_q, xpu_k, xpu_v = fusedQKVGEMM(input, weight_q, weight_k, weight_v)
    print(np.allclose(xpu_q, cpu_q, atol=1e-3, rtol=1e-3))
    print(np.allclose(xpu_k, cpu_k, atol=1e-3, rtol=1e-3))
    print(np.allclose(xpu_v, cpu_v, atol=1e-3, rtol=1e-3))

if __name__ == '__main__':
   testQKVFusion()