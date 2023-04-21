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
'''Init file for register XPU backend'''

from jax._src import xla_bridge as xla_bridge
from jax._src.lib import xla_client

import os

def make_factory(name, path):
  def factory():
    xla_client.load_pjrt_plugin_dynamically(name, path)
    return xla_client.make_c_api_client(name)
  return factory

# PATH=path-of-so ## search .so based on py path
dir_path = os.path.dirname(__file__)
lib_path = os.path.join(dir_path, "libitex_xla_extension.so")
xla_bridge.register_backend_factory(
  "xpu", make_factory("xpu", lib_path), priority=400
)
