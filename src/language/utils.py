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

import re


def camel_case_to_underscore(name):
    """convert string from 'AaaBbbbCccDdd' -> 'Aaa_Bbbb_Ccc_Ddd'"""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    typename = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
    return typename


def to_snake_case(name):
    """convert string from 'AaaBbbbCccDdd' -> 'aaa_bbbb_ccc_ddd'"""
    return camel_case_to_underscore(name).lower()
