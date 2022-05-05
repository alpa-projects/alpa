# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

# -- Project information -----------------------------------------------------

project = 'Alpa'
#copyright = '2022, <Author>'
#author = '<Author>'

# The full version, including alpha/beta/rc tags
release = '0.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx_gallery.gen_gallery',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    ]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# Explicitly define the order within a subsection.
# The listed files are sorted according to the list.
# The unlisted files are sorted by filenames.
# The unlisted files always appear after listed files.
within_subsection_order = {
    "tutorials": [
        "getting_started.py",
        "pipeshard_parallelism.py",
        "alpa_vs_pmap.py",
        "advanced_api_usage.py",
    ],
}

class WithinSubsectionOrder:
    def __init__(self, src_dir):
        self.src_dir = src_dir.split("/")[-1]

    def __call__(self, filename):
        # If the order is provided, use the provided order
        if (
            self.src_dir in within_subsection_order
            and filename in within_subsection_order[self.src_dir]
        ):
            index = within_subsection_order[self.src_dir].index(filename)
            assert index < 1e10
            return "\0%010d" % index

        # Otherwise, sort by filename
        return filename


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# sphinx-gallery configuration
sphinx_gallery_conf = {
    'examples_dirs': ['gallery/tutorials'],
    'gallery_dirs': ['tutorials'],
    'within_subsection_order': WithinSubsectionOrder,
    'backreferences_dir': 'gen_modules/backreferences',
    "filename_pattern": os.environ.get("ALPA_TUTORIAL_EXEC_PATTERN", r"(?<!pipeshard_parallelism).py"),
}

# configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/{.major}'.format(sys.version_info), None),
    'matplotlib': ('https://matplotlib.org/', None),
    'pandas': ('https://pandas.pydata.org/', None),
}

# -- Monkey patch -------------------------------------------------

# Fix a bug in sphinx_gallery
from sphinx_gallery import gen_rst
setattr(gen_rst._LoggingTee, "close", lambda x:x.restore_std())
