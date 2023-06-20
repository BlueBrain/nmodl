# Copyright 2023 Blue Brain Project, EPFL.
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

class Argument:
    """Utility class for holding all arguments for node classes"""

    def __init__(self):
        # BaseNode
        self.class_name = ""
        self.nmodl_name = ""
        self.prefix = ""
        self.suffix = ""
        self.force_prefix = ""
        self.force_suffix = ""
        self.separator = ""
        self.brief = ""
        self.description = ""

        # ChildNode
        self.typename = ""
        self.varname = ""
        self.is_public = False
        self.is_vector = False
        self.is_optional = False
        self.add_method = False
        self.get_node_name = False
        self.getter_method = False
        self.getter_override = False

        # Node
        self.base_class = ""
        self.has_token = False
        self.url = None
