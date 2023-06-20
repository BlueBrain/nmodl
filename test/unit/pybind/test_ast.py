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

from nmodl.dsl import ast
import nmodl.dsl as nmodl
import pytest

class TestAst(object):
    def test_empty_program(self):
        pnode = ast.Program()
        assert str(pnode) == ''

    def test_ast_construction(self):
        string = ast.String("tau")
        name = ast.Name(string)
        assert nmodl.to_nmodl(name) == 'tau'

        int_macro = nmodl.ast.Integer(1, ast.Name(ast.String("x")))
        assert nmodl.to_nmodl(int_macro) == 'x'

        statements = []
        block = ast.StatementBlock(statements)
        neuron_block = ast.NeuronBlock(block)
        assert nmodl.to_nmodl(neuron_block) == 'NEURON {\n}'

    def test_get_parent(self):
        x_name = ast.Name(ast.String("x"))
        int_macro = nmodl.ast.Integer(1, x_name)
        assert x_name.parent == int_macro # getting the parent

    def test_set_parent(self):
        x_name = ast.Name(ast.String("x"))
        y_name = ast.Name(ast.String("y"))
        int_macro = nmodl.ast.Integer(1, x_name)
        y_name.parent = int_macro # setting the parent
        int_macro.macro = y_name
        assert nmodl.to_nmodl(int_macro) == 'y'

    def test_ast_node_repr(self):
        string = ast.String("tau")
        name = ast.Name(string)
        assert repr(name) == nmodl.to_json(name, compact=True)

    def test_ast_node_str(self):
        string = ast.String("tau")
        name = ast.Name(string)
        assert str(name) == nmodl.to_nmodl(name)
