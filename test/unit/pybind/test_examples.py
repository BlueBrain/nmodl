from json import loads
from pathlib import Path

from nmodl import dsl


def test_example():
    """
    Test for the Python API from example
    """

    examples = dsl.list_examples()

    # ordering may be off so we use a set
    assert set(examples) == {"exp2syn.mod", "expsyn.mod", "hh.mod", "passive.mod"}

    driver = dsl.NmodlDriver()
    for example in examples:
        nmodl_string = dsl.load_example(example)
        modast = driver.parse_string(nmodl_string)
        assert loads(dsl.to_json(modast)) == loads(
            (
                Path(__file__).parent / "data" / example.replace(".mod", ".json")
            ).read_text()
        )
