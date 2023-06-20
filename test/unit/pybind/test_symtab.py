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

import io
from nmodl.dsl import ast, visitor, symtab

def test_symtab(ch_ast):
    v = symtab.SymtabVisitor()
    v.visit_program(ch_ast)
    s = ch_ast.get_symbol_table()
    m = s.lookup('m')
    assert m is not None
    assert m.get_name() == "m"
    assert m.has_all_properties(symtab.NmodlType.state_var | symtab.NmodlType.prime_name) is True

    mInf = s.lookup('mInf')
    assert mInf is not None
    assert mInf.get_name() == "mInf"
    assert mInf.has_any_property(symtab.NmodlType.range_var) is True
    assert mInf.has_any_property(symtab.NmodlType.local_var) is False

    variables = s.get_variables_with_properties(symtab.NmodlType.range_var, True)
    assert len(variables) == 2
    assert str(variables[0]) == "mInf [Properties : range assigned_definition]"


def test_visitor_python_io(ch_ast):
    lookup_visitor = visitor.AstLookupVisitor()
    eqs = lookup_visitor.lookup(ch_ast, ast.AstNodeType.DIFF_EQ_EXPRESSION)
    stream = io.StringIO()
    pv = visitor.NmodlPrintVisitor(stream)
    eqs[0].accept(pv)
    #  `del pv` is not required to have the ostream flushed
    assert stream.getvalue() == "m' = mInf-m"
