# ***********************************************************************
# Copyright (C) 2018-2019 Blue Brain Project
#
# This file is part of NMODL distributed under the terms of the GNU
# Lesser General Public License. See top-level LICENSE file for details.
# ***********************************************************************

import re


def camel_case_to_underscore(name):
    """convert string from 'AaaBbbbCccDdd' -> 'Aaa_Bbbb_Ccc_Ddd'"""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    typename = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
    return typename


def to_snake_case(name):
    """convert string from 'AaaBbbbCccDdd' -> 'aaa_bbbb_ccc_ddd'"""
    return camel_case_to_underscore(name).lower()


def check_override(node_name, signature, node_members_signatures):
    add = ""
    if node_name in node_members_signatures :
        # print(node_members_signatures[node_name])
        if signature in node_members_signatures[node_name]:

            add = " override "
        else:
            node_members_signatures[node_name].add(signature)
    else :
        node_members_signatures[node_name] = {signature}

    return signature + add

def inherit_signatures(child, parent, node_members_signatures):
    if parent in node_members_signatures :
        node_members_signatures[child] = node_members_signatures[parent]