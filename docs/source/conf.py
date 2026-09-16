# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))  # points at repo root


# -- Project information -----------------------------------------------------

project = 'YRoots'
copyright = '2023-2026, BYU Math'
author = 'BYU Math'
root_doc = "index"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.coverage', 'sphinx.ext.napoleon'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'yroots'
html_theme_path = ['_theme']
html_title = 'YRoots documentation'
# Built by GitHub Actions, GITHUB_REPOSITORY names the repository publishing
# the site, so these links follow this repository wherever it lives -- here or
# upstream -- without being edited. The sibling repositories keep their names
# under either owner; only the owner differs. The fallback keeps a local build
# pointing somewhere real.
_repository = os.environ.get('GITHUB_REPOSITORY', 'wlgns0330/Rootfinding-serial')
_owner = _repository.split('/')[0]

html_theme_options = {
    # This repository publishes only the API docs, so the header points back at
    # the project's landing page, which the parallel repository's site carries.
    'landing_url': f'https://{_owner}.github.io/RootFinding/',
    'repo_url': f'https://github.com/{_repository}',
    'sibling_url': f'https://github.com/{_owner}/RootFinding',
    'sibling_label': 'yroots (parallel)',
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []
