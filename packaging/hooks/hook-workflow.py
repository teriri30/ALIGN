"""The repository's workflow.py is an application module, not a distribution.

PyInstaller's contrib hook for the unrelated PyPI package named ``workflow``
tries to copy package metadata and collides with this project's module name.
The application module is discovered from the normal analysis path, so this
project-local hook intentionally contributes no extra data.
"""
