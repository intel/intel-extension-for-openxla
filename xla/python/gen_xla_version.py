# Copyright (c) 2021-2024 Intel Corporation
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

import os
import sys


def parse_args(argv):
    result = {}
    for arg in argv:
        k, v = arg.split("=")
        result[k] = v

    return result


def generate_version(header_in, hash_value, header_out):
    with open(os.path.expanduser(header_in), encoding='utf-8') as inf:
        content = inf.read()
        content = content.replace("@VERSION_HASH@", hash_value)

        header_out = os.path.expanduser(header_out)
        header_out_dir = os.path.dirname(header_out)
        if not os.path.exists(header_out_dir):
            os.makedirs(header_out_dir, exist_ok=True)

        with open(header_out, "w",  encoding='utf-8') as outf:
            outf.write(content)


def main():
    args = parse_args(sys.argv[1:])
    generate_version(args["--in"], args["--hash"], args["--out"])


if __name__ == "__main__":
    main()
