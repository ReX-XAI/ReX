# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "ReX"
copyright = "2024, David Kelly"
author = "David Kelly, Liz Ing-Simmons and other contributors"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "autoapi.extension",
    "sphinx.ext.autodoc.typehints",
    "myst_parser",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    'sphinxarg.ext',
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_sidebars = {
    '**': [
        'about.html',
        'searchfield.html',
        'navigation.html',
        'relations.html',
        'donate.html',
    ]
}
html_static_path = ["_static"]
html_theme_options = {
   "logo": "rex_logo.png"
}

# -- AutoAPI -----------------------------------------------------------------
# https://sphinx-autoapi.readthedocs.io/en/latest/
autoapi_dirs = ["../rex_xai/"]
autodoc_typehints = "description"

# -- Intersphinx --------------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://pytorch.org/docs/stable/',None),
    'sqla': ('https://docs.sqlalchemy.org/en/latest/', None)
    }

# -- MyST --------------------------------------------------------------
# https://myst-parser.readthedocs.io/en/latest/
myst_enable_extensions = [
    "attrs_inline"
]
