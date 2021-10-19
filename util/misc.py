"""
misc utils

Reference: https://github.com/jayelm/compexp/blob/master/vision/util/misc.py
"""


def safe_layername(layer):
    if isinstance(layer, list):
        return "-".join(map(str, layer))
    else:
        return layer
