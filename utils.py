# -*- coding: utf-8 -*
"""
"""

import functools


def lazy_property(func):
    """
    Decorator function to make a property lazy.
    A lazy property is one only evaluated once (evaluated only on first time call)
    :param func:
    :return:
    """
    attribute = '_lazy_' + func.__name__

    @property
    @functools.wraps(func)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, func(self))
        return getattr(self, attribute)

    return wrapper
