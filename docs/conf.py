"""Sphinx configuration."""
project = "PyTorch IE"
author = "Christoph Alt"
copyright = "2022, Christoph Alt"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "furo"
