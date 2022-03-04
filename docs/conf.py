"""Sphinx configuration."""
project = "PyTorch IE"
author = "Christoph Alt"
copyright = "2022, Christoph Alt"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]
autodoc_typehints = "description"
html_theme = "furo"
