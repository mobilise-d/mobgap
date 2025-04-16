(dev_guide)=
# Development Guide

This document contains information for developers that need further in-depth information on how to setup and use tools
and learn about programming methods used in development of this project.

If you are looking for a higher level overview over the guiding ideas and structure of this project, please visit the
[Project Structure document](#proj_struct).

## Project Setup and UV

### Repo cloning

To clone the repository, you can use the following command:

```bash
git clone https://github.com/mobilise-d/mobgap.git
cd mobgap
```
**In case you encounter any issues (with this command or any command below), please check the section on
 [trouble shooting](#trouble-shooting)**.

### Setup Python

*mobgap* uses `uv` as a Python version and dependency manager.
`uv` can handle both the installation of Python and any dependencies.

Install `uv` following the instructions found [here](https://docs.astral.sh/uv/getting-started/installation/).
Once you have uv installed, open a terminal and run `uv --help` to check if it is running correctly.

Then navigate to the project directory and run the following commands:

```bash
uv python pin 3.11
uv sync
```
This will create a new virtual environment with Python 3.11 and install all dependencies listed in `pyproject.toml`.

If you want a different python version for testing, rerun these two commands with a different version.

For more tips for using `uv` check out [this guide](https://github.com/mad-lab-fau/mad-cookiecutter/blob/main/python-setup-tips.md)


## Tools we are using

To make it easier to run commandline tasks we use [poethepoet](https://github.com/nat-n/poethepoet) to provide a 
cross-platform cli for common tasks.

To list the available tasks, run:

```bash
$ uv run poe
...
CONFIGURED TASKS
  format            
  format_unsafe     
  lint              Lint all files with ruff.
  ci_check          Check all potential format and linting issues.
  test              Run Pytest with coverage.
  docs              Build the html docs using Sphinx.
  docs_clean        Remove all old build files and build a clean version of the docs.
  docs_linkcheck    Check all links in the built html docs.
  docs_preview      Preview the built html docs.
  version           Bump the version number in all relevant files.
  conf_jupyter      Add a new jupyter kernel for the project.
  remove_jupyter    Remove the project specific jupyter kernel.

```

To run one of the commands execute (e.g. the `test` command):
```bash
uv run poe test
```

**Protip**: If you installed poethepoet globally, you can skip the `uv run` part at the beginning.

### Formatting and Linting

To ensure that the whole library uses a consistent **format**, we use [ruff](https://github.com/astral-sh/ruff) to autoformat our code.
Beyond automatically fixing issues, we use *ruff* to handle all other **linting** tasks.

For **documentation** we follow the numpy doc-string guidelines and autobuild our API documentation using *Sphinx*.
To make your live easier, you should also set your IDE tools to support the numpy docstring conventions.

To run formatting you can use

```bash
uv run poe format
```

and for linting you can run

```bash
uv run poe lint
```

Tou should run this as often as possible!
At least once before any `git push`.


### Building Documentation

To build the documentation, run:

```bash
uv run poe docs
```

Afterwards, you can preview the documentation by running:

```bash
uv run poe docs_preview
```

This should display a URL in your terminal, which you can open in your browser.

Sometimes building the documentation fails, because you still have some old files around.
In this case, you can run:

```bash
uv run poe docs_clean
```

to force a clean build.
Note, that this might take substantially longer than a normal build.


### Testing and Test data

This library uses `pytest` for **testing**. Besides using the poe-command, you can also use an IDE integration
available for most IDEs.
From the general structure, each file has a corresponding `test_...` file within a similar sub structure.

#### Common Tests

For basically all new algorithms we want to test a set of basic functionalities.
For this we have `tpcp.testing.TestAlgorithmMixin`.
To use the general mixin, create a new test class, specify the `algorithm_class`, the `after_action_instance` fixture
and set `__test__ = True`.
For more details see the docstring of the mixin.

#### Test Data

Test data is available in the `example_data` folder.
Within scripts or examples, the recommended way to access it is using the functions in `mobgap.example_data`.

TODO: Update this section once we know how to handle example data
```python
from mobgap.example_data import get_healthy_example_imu_data

```

Within tests you can also use the pytest fixtures defined `tests/conftest.py`.

```python
# Without import in any valid test file

def test_myfunc(healthy_example_imu_data):
    ...
```

#### Testing Examples

For each mature feature their should also be a corresponding example in the `examples` folder.
To make sure they work as expected, we also test them using `pytest`.
For this create a new test function in `tests/test_examples/` and simply import the example
within the respective function.
This will execute the example and gives you access to the variables defined in the example.
They can then be tested.
Most of the time a regression/snapshot test is sufficient (see below).

#### Snapshot Testing

To prevent unintentional changes to the data, this project makes use of regression tests.
These tests store the output of a function and compare the output of the same function at a later time to the stored
information.
This helps to ensure that a change did not modify a function unintentionally.
To make this easy, this library contains a small PyTest helper to perform regression tests.

A simple regression test looks like this:

```python
import pandas as pd

def test_regression(snapshot):
    # Do my tests
    result_dataframe = pd.DataFrame(...)
    snapshot.assert_match(result_dataframe)
```

This test will store `result_dataframe` in a json file if the test is run for the first time.
At a later time, the dataframe is loaded from this file to compare it.
If the new `result_dataframe` is different from the file content the test fails.

In case the test fails, the results need to be manually reviewed.
If the changes were intentionally, the stored data can be updated by either deleting, the old file
and rerunning the test, or by running `pytest --snapshot-update`. Be careful, this will update all snapshots.

The results of a snapshot test should be committed to the repo.
Make reasonable decisions when it comes to the datasize of this data.

For more information see `tpcp.testing.PyTestSnapshotTest`.
 
## Configure your IDE
(Configure-your-IDE)=

### Pycharm

**Test runner**: Set the default testrunner to `pytest`. 

**Autoreload for the Python console**:

You can instruct Pycharm to automatically reload modules upon changing by adding the following lines to
settings->Build,Excecution,Deployment->Console->Python Console in the Starting Script:

```python
%load_ext autoreload
%autoreload 2
```

### Jupyter Lab/Notebooks

While we do not (and will not) use Jupyter Notebooks in mobgap, it might still be helpful to use Jupyter to debug and
prototype your scientific code.
To set up a Jupyter environment that has mobgap and all dependencies installed, run the following commands:

```bash
uv run poe conf_jupyter
``` 

After this you can start Jupyter as always, but select "mobgap" as a kernel when you want to run a notebook.

Remember to use the autoreload extension to make sure that Jupyter reloads mobgap, when ever you change something in 
the library.
Put this in your first cell of every Jupyter Notebook to activate it:

```python
%load_ext autoreload  # Load the extension
%autoreload 2  # Autoreload all modules
```

## Release Model

mobgap follows typically semantic visioning: A.B.C (e.g. 1.3.5)

- `A` is the major version, which will be updated once there were fundamental changes to the project
- `B` is the minor version, which will be updated whenever new features are added
- `C` is the patch version, which will be updated for bugfixes

As long as no new minor or major version is released, all changes should be interface compatible.
This means that the user can update to a new patch version without changing any user code!

This means at any given time we need to support and work with two versions:
The last minor release, which will get further patch releases until its end of life.
The upcoming minor release for which new features are developed at the moment.
However, in most cases we will also not create proper patch releases, but expect users to update to the newest git 
version, unless it was an important and major bug that got fixed.

Note that we will not support old minor releases after the release of the next minor release to keep things simple.
We expect users to update to the new minor release, if they want to get new features and bugfixes.

To make such an update model go smoothly for all users, we keep an active changelog, that should be modified a feature
is merged or a bug fixed.
In particular changes that require updates to feature code should be prominently highlighted in the "Migration Guide"
section.

There is no fixed timeline for a release, but rather a list of features we will plan to include in every release.
Releases can happen often and even with small added features.

If you are ready to release a new version, you can use the following command:

```bash
uv run poe version {major|minor|patch}
```

This will update the version number in all relevant files.
Then head over to the ChangeLog and update the "Unreleased" section to the new version number and set the release date.

Then follow the following steps:

- Double-check the changelog and the migration guide
- Commit and push all changes including the changed version number
- **Wait for the CI to finish**
- Create a new release on GitHub (you can create a new tag directly from the same interface) and add the changelog to 
  the release notes
- Wait for the "publish" job to finish -> Double-check the PyPI page if the new version is available

## Environment Variables during development

Some aspects of mobgap (in particular in the examples, scripts, and revalidation files) are controlled by environment 
variables as settings need to be specific to the user or are path references to data files that need to be downloaded
independently.
Environment variables are usually only required for developers, or for users that want to rerun the examples without
modified them (however, for them it is usually easier to just modify the example).

To avoid needing to set these variables every time, developers can create a `.env` file in the root of the project.
When an environment variable is read by using `mobgap.utils.misc.get_env_variable`, it will first check if the variable
is set in the environment and if not, the value from `.env` file will be used.
A list of variables with their explanations that are used in the project can be found in `.env.template` in the root folder.

The `.env` file is ignored by git, and should not be committed to the repository.

When working with values returned by `mobgap.utils.misc.get_env_variable`, keep in mind that they are always strings.
You need to handle parsing accordingly.
In most cases, empty string, or `0` is used to indicate that the variable is not set.

## Git Workflow

As multiple people are expected to work on the project at the same time, we need a proper git workflow to prevent issues.

### Branching structure

This project uses a main + feature branches.
This workflow is well explained [here](https://www.atlassian.com/blog/git/simple-git-workflow-is-simple).
  
All changes to the main branch should be performed using feature branches.
Before merging, the feature branches should be rebased onto the current main.

Remember, Feature branchs...:

- should be short-lived
- should be dedicated to a single feature
- should be worked on by a single person
- must be merged via a Pull Request and not manually
- must be reviewed before merging
- must pass the pipeline checks before merging
- should be rebased onto main if possible (remember only rebase if you are the only person working on this branch!)
- should be pushed soon and often to allow everyone to see what you are working on
- should be associated with a merge request, which is used for discussions and code review.
- that are not ready to review, should have a merge request prefixed with `WIP: `
- should also close issues that they solve, once they are merged

Workflow

```bash
# Create a new branch
git switch main
git pull origin main
git switch -c new-branch-name
git push origin new-branch-name
# Go to Gitlab and create a new Merge Request with WIP prefix

# Do your work
git push origin new-branch-name

# In case there are important changes in main, rebase
git fetch origin main
git rebase origin/main
# resolve potential conflicts
git push origin new-branch-name --force-with-lease

# Create a merge request and merge via web interface

# Once branch is merged, delete it locally, start a new branch
git switch main
git branch -D new-branch-name

# Start at top!
```

### For large features

When implementing large features it sometimes makes sense to split it into individual pull requests/sub-features.
If each of these features are useful on their own, they should be merged directly into main.
If the large feature requires multiple merge requests to be usable, it might make sense to create a long-lived feature
branch, from which new branches for the sub-features can be created.
It will act as a develop branch for just this feature.
Remember, to rebase this temporary dev branch onto master from time to time.

.. note:: Due to the way mobgap is build, it is often possible to develop new features (e.g. algorithms) without
          touching the mobgap source code.
          Hence, it is recommended to devlop large features in a separate repository and only merge them into mobgap
          once you worked out all the kinks.
          This avoids long living feature branches in mobgap and allows you to develop your feature in a more flexible 
          way.

### General Git Tips

- Communicate with your Co-developers
- Commit often
- Commit in logical chunks
- Don't commit temp files
- Write at least a somewhat [proper messages](https://chris.beams.io/posts/git-commit/)
   - Use the imperative mood in the subject line
   - Use the body to explain what and why vs. how
   - ...more see link above


## Trouble Shooting
(trouble-shooting)=

### "Unable to create file ... File name too long" on Windows during git clone

This is a known issue with Windows and git.
We try to avoid this by keeping our filenames short and not nesting too deep.
So if you encounter this issue, please let us know, so we can fix it.

However, it might be a good idea to also change the config on your end to support longer filenames.
To do this, run the following command in your git bash:

```bash
git config --global core.longpaths true
```

Learn more about this here: https://stackoverflow.com/questions/22575662/filename-too-long-in-git-for-windows

## Other important information

### Sklearn Versions and InconsistentVersionWarning

Scikit learn model are not guaranteed to work across sklearn versions.
The laterality algorithms that we use ships a pretrained model that is generated by the `scripts/retrain_lrc_ullrich_msproject.py` script.
This script uses the sklearn version that is installed in the environment.
In case, users use the model with a different sklearn version, they might encounter a `InconsistentVersionWarning` when they import the respective
module.

This is not generally a problem, but from time to time, we should retrain the model with the latest sklearn version
and then update the minimum sklearn version that we support in the pyproject.toml file.

This is not ideal... But there is no real way of packaging models that are version independent.