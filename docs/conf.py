# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys

sys.path.insert(0, os.path.abspath("../src/bayesopt4ros"))
sys.path.insert(0, os.path.abspath("../nodes"))
sys.path.insert(0, os.path.abspath("../test/integration"))


# -- Project information -----------------------------------------------------

project = "BayesOpt4ROS"
copyright = "2021, Lukas Froehlich"
author = "Lukas Froehlich"


# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx.ext.autosectionlabel",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Miscealaneous -----------------------------------------------------------
todo_include_todos = True
today_fmt = "%A %d %B %Y, %X"

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"

html_static_path = ["_static"]
