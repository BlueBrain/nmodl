from ._nmodl import *
from pkg_resources import *

def example_dir():
    """Returns directory containing NMODL examples

    NMODL Framework is installed with few sample example of
    channels. This method can be used to get the directory
    containing all mod files.

    Returns:
        Full path of directory containing sample mod files
    """
    resource =  "share/example"
    if resource_exists(__name__, resource) and resource_isdir(__name__, resource):
        return resource_string(__name__, resource)
    else:
        raise FileNotFoundError("Could nto find sample mod files")
