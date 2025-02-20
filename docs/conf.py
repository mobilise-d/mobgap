# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import re
import sys
from datetime import datetime
from pathlib import Path

import toml
from sphinx_gallery.sorting import ExplicitOrder, FileNameSortKey

HERE = Path(__file__)

sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE.parent / "_ext"))


def convert_github_links(base_url, text):
    regex = base_url + r"(pull|issues|commit)/(\w+)"

    def substitute(matchobj) -> str:
        if matchobj.group(1) == "commit":
            return f"[{matchobj.group(2)[:5]}]({matchobj.group(0)})"
        return f"[#{matchobj.group(2)}]({matchobj.group(0)})"

    return re.sub(regex, substitute, text)


def convert_github_md_admonitions(text):
    """Converts the GitHub style admonitions to the myst style."""

    def substitute(matchobj) -> str:
        admonition_type = matchobj.group(1).lower()
        text = re.sub(r"\n>", "\n", matchobj.group(2)).strip()

        return f"```{{{admonition_type}}}\n{text}\n```\n"

    return re.sub(r"> \[!([^\]]+)\]((?:\n>[^\n]+)+)", substitute, text)


# -- Project information -----------------------------------------------------

URL = "https://github.com/mobilise-d/mobgap/"
# Info from poetry config:
info = toml.load("../pyproject.toml")["project"]

project = info["name"]
author = ", ".join(f"{author['name']} <{author['email']}>" for author in info["authors"])
release = info["version"]

copyright = f"2023 - {datetime.now().year}, MaD Lab, FAU in the name of the Mobilise-D consortium"

# -- Copy the README and Changelog and fix image path --------------------------------------
HERE = Path(__file__).parent
with (HERE.parent / "README.md").open() as f:
    out = f.read()
out = convert_github_links(URL, out)
out = convert_github_md_admonitions(out)
out = out.replace("./LICENSE", URL + "/blob/main/LICENSE")
out = out.replace("./NOTICE", URL + "/blob/main/NOTICE")
out = out.replace("./docs/_static/", "./_static/")
with (HERE / "README.md").open("w+") as f:
    f.write(out)

with (HERE.parent / "CHANGELOG.md").open() as f:
    out = f.read()
out = convert_github_links(URL, out)
with (HERE / "CHANGELOG.md").open("w+") as f:
    f.write(out)

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "numpydoc",
    "sphinx.ext.linkcode",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    # "sphinx.ext.imgconverter",
    "sphinx_gallery.gen_gallery",
    "myst_parser",
    "fullscreen_gallery",
]

# Taken from sklearn config
# For maths, use mathjax by default and svg if NO_MATHJAX env variable is set
# (useful for viewing the doc offline)
if os.environ.get("NO_MATHJAX"):
    extensions.append("sphinx.ext.imgmath")
    imgmath_image_format = "svg"
    mathjax_path = ""
else:
    extensions.append("sphinx.ext.mathjax")
    mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"

# this is needed for some reason...
# see https://github.com/numpy/numpydoc/issues/69
numpydoc_class_members_toctree = False

autodoc_default_options = {"members": True, "inherited-members": True, "special_members": True}
autodoc_typehints = "signature"

python_maximum_signature_line_length = 88

# Add any paths that contain templates here, relative to this directory.
templates_path = ["templates"]

# generate autosummary even if no references
autosummary_generate = True
autosummary_generate_overwrite = True

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "templates"]

# The reST default role (used for this markup: `text`) to use for all
# documents.
default_role = "literal"

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# Activate the theme.
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": URL,
    "show_prev_next": False,
}
html_context = {
    "github_user": "mobilise-d",
    "github_repo": "mobgap",
    "github_version": "main",
    "doc_path": "docs",
}

html_favicon = "_static/logo/mobilise_d_logo.ico"
html_logo = "_static/logo/mobilise_d_logo.png"


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = [
    "custom_css/pandas.css",
]
# -- Options for extensions --------------------------------------------------
# Intersphinx

# intersphinx configuration
intersphinx_module_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "tpcp": ("https://tpcp.readthedocs.io/en/latest/", None),
    "pywt": ("https://pywavelets.readthedocs.io/en/latest/", None),
}

user_agent = "Mozilla/5.0 (X11; Linux x86_64; rv:25.0) Gecko/20100101 Firefox/25.0"

intersphinx_mapping = {
    "python": (f"https://docs.python.org/{sys.version_info.major}", None),
    **intersphinx_module_mapping,
}

# Sphinx Gallary
sphinx_gallery_conf = {
    "examples_dirs": ["../examples", "../revalidation"],
    "gallery_dirs": ["./auto_examples", "./auto_revalidation"],
    "reference_url": {"mobgap": None},
    # 'default_thumb_file': 'fig/logo.png',
    "backreferences_dir": "modules/generated/backreferences",
    "doc_module": ("mobgap",),
    "filename_pattern": r"^(?!.*_no_exc\.py$).*\.py$",  # ignore files with _no_exc
    "remove_config_comments": True,
    "show_memory": True,
    "subsection_order": ExplicitOrder(
        [
            "../examples/data",
            "../examples/pipeline",
            "../examples/gait_sequences",
            "../examples/initial_contacts",
            "../examples/laterality",
            "../examples/cadence",
            "../examples/stride_length",
            "../examples/turning",
            "../examples/wba",
            "../examples/aggregation",
            "../examples/data_transform",
            "../examples/dev_guides",
            "../revalidation/full_pipeline",
            "../revalidation/full_pipeline",
            "../revalidation/gait_sequences",
            "../revalidation/initial_contacts",
            "../revalidation/laterality",
            "../revalidation/cadence",
            "../revalidation/stride_length",
        ]
    ),
    "within_subsection_order": FileNameSortKey,
}


from sphinxext.githublink import make_linkcode_resolve

linkcode_resolve = make_linkcode_resolve(
    "mobgap",
    "https://github.com/mobilise-d/mobgap/blob/{revision}/{package}/{path}#L{lineno}",
)
