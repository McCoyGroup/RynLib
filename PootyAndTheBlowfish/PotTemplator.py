"""
Defines classes to take a potential .so file an create a python extension that can wrap it and call into the potential
"""

from .Templator import TemplateWriter

__all__ = [
    "PotentialTemplate"
]


class PotentialTemplate(TemplateWriter):
    ...