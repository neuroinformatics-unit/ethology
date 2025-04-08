# Configuration file for the Sphinx documentation builder.
"""Sphinx configuration for ethology documentation."""

import os
import sys

import setuptools_scm

# Used when building API docs, put the dependencies
# of any class you are documenting here
autodoc_mock_imports: list[str] = ["cv2", "torch"]

# Add the module path to sys.path here.
# If the directory is relative to the documentation root,
# use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath("../.."))

project = "ethology"
copyright = "2024, University College London"
author = "Adam Tyson"
try:
    release = setuptools_scm.get_version(root="../..", relative_to=__file__)
    release = release.split("+")[0]  # remove git hash
except LookupError:
    # if git is not initialised, still allow local build
    # with a dummy version
    release = "0.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "sphinx_autodoc_typehints",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "nbsphinx",
    "notfound.extension",
    "sphinx_design",
    "sphinx_sitemap",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.todo",
    "sphinx.ext.graphviz",
    "sphinx.ext.ifconfig",
    "sphinx.ext.mathjax",
    "sphinx.ext.imgmath",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.term",
]

# Configure the myst parser to enable cool markdown features
# See https://sphinx-design.readthedocs.io
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "tasklist",
]
# Automatically add anchors to markdown headings
myst_heading_anchors = 4

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# Automatically generate stub pages for API
autosummary_generate = True
autodoc_default_flags = ["members", "inherited-members"]

# Prefix section labels with the document name
autosectionlabel_prefix_document = True

# -- Options for HTML output -------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = "ethology"

# Customize the theme
html_theme_options = {
    "icon_links": [
        {
            # Label for this link
            "name": "GitHub",
            # URL where the link will redirect
            "url": "https://github.com/neuroinformatics-unit/ethology",
            # Icon class (if "type": "fontawesome"),
            # or path to local image (if "type": "local")
            "icon": "fa-brands fa-github",
            # The type of image to be used (see below for details)
            "type": "fontawesome",
        },
        {
            "name": "Zulip (chat)",
            "url": "https://neuroinformatics.zulipchat.com/#narrow/channel/483869-Ethology",
            "icon": "fa-solid fa-comments",
            "type": "fontawesome",
        },
    ],
    "logo": {
        "text": f"{project} v{release}",
    },
    "footer_start": ["footer_start"],
    "footer_end": ["footer_end"],
    "external_links": [],
}

# Redirect the webpage to another URL
# Sphinx will create the appropriate CNAME file in the build directory
# The default is the URL of the GitHub pages
# https://www.sphinx-doc.org/en/master/usage/extensions/githubpages.html
github_user = "neuroinformatics-unit"
html_baseurl = "https://ethology.neuroinformatics.dev"
sitemap_url_scheme = "{link}"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = [
    ("css/custom.css", {"priority": 100}),
]
html_favicon = "_static/light-logo-niu.png"

# The linkcheck builder will skip verifying that anchors exist when checking
# these URLs
linkcheck_anchors_ignore_for_url = [
    "https://neuroinformatics.zulipchat.com/",
    "https://gin.g-node.org/G-Node/Info/wiki/",
]
# A list of regular expressions that match URIs that should not be checked
linkcheck_ignore = [
    "https://opensource.org/license/bsd-3-clause/",  # to avoid odd 403 error
]


myst_url_schemes = {
    "http": None,
    "https": None,
    "ftp": None,
    "mailto": None,
    "ethology-github": "https://github.com/neuroinformatics-unit/ethology/{{path}}",
    "ethology-zulip": "https://neuroinformatics.zulipchat.com/#narrow/channel/483869-Ethology",
    "setuptools-scm": "https://setuptools-scm.readthedocs.io/en/latest/{{path}}#{{fragment}}",
    "sphinx-doc": "https://www.sphinx-doc.org/en/master/usage/{{path}}#{{fragment}}",
    "conda": "https://docs.conda.io/en/latest/",
    "gin": "https://gin.g-node.org/{{path}}#{{fragment}}",
    "github-docs": "https://docs.github.com/en/{{path}}#{{fragment}}",
    "mamba": "https://mamba.readthedocs.io/en/latest/",
    "myst-parser": "https://myst-parser.readthedocs.io/en/latest/{{path}}#{{fragment}}",
    "napari": "https://napari.org/dev/{{path}}",
}


# What to show on the 404 page
notfound_context = {
    "title": "Page Not Found",
    "body": """
<h1>Page Not Found</h1>

<p>Sorry, we couldn't find that page.</p>

<p>Try using the search box or go to the homepage.</p>

<p>You can also email us on code@adamltyson.com.</p>

""",
}

# needed for GH pages (vs readthedocs),
# because we have no '/<language>/<version>/' in the URL
notfound_urls_prefix = None
