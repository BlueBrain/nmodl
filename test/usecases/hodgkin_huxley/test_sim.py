import shutil
from pathlib import Path

import numpy as np
import pytest
from neuron import gui
from neuron import h
from neuron import load_mechanisms
from neuron.units import ms
from usecases import compile_modfile


@pytest.mark.parametrize(
    "compiler_path",
    [
        None,
        shutil.which("nmodl"),
    ],
)
def test_hodhux(compiler_path):
    """
    Test for Hodkin-Huxley model.
    """
    compile_modfile(Path(__file__).parent / "hodhux.mod", compiler_path)
    load_mechanisms(Path(__file__).parent)
    nseg = 1

    s = h.Section()
    s.insert("hodhux")
    ic = h.IClamp(s(0.5))
    ic.delay = 0
    ic.dur = 1e9
    ic.amp = 10
    s.nseg = nseg

    v_hoc = h.Vector().record(s(0.5)._ref_v)
    t_hoc = h.Vector().record(h._ref_t)

    h.stdinit()
    h.tstop = 100.0 * ms
    h.run()

    v = np.array(v_hoc.as_numpy())
    t = np.array(t_hoc.as_numpy())

    v_l1 = np.sum(np.abs(v)) / v.size
    v_tv = np.sum(np.abs(np.diff(v)))

    # (Nearly) exact values of the L1 and TV (semi-)norms, computed with
    # `dt = 0.0002, 0.00005`.
    v_l1_exact = 58.673
    v_tv_exact = 1270.8

    assert abs(v_l1 - v_l1_exact) < 0.004 * v_l1_exact
    assert abs(v_l1 - v_l1_exact) < 0.01 * v_tv_exact
