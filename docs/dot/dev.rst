**************
For developers
**************

Contributing
------------

``dot`` is open source and built on open source, and we'd love to have your
contributions to the software.

To make a code contribution for the first time, please follow these
`delightfully detailed instructions from astropy
<https://docs.astropy.org/en/stable/development/workflow/development_workflow.html>`_.

For coding style guidelines, we also point you to the
`astropy style guide <https://docs.astropy.org/en/stable/development/codeguide.html>`_.

Building the docs
^^^^^^^^^^^^^^^^^

You can check that your documentation builds successfully by building the docs
locally. Run::

    pip install tox
    tox -e build_docs

Testing
^^^^^^^

You can check that your new code doesn't break the old code by running the tests
locally. Run::

    tox -e test


Releasing dot
^^^^^^^^^^^^^

Here are some quick notes on releasing dot.

The astropy package template that dot is built on requires the following
steps to prepare a release. First you need to clean up the repo before you
release it.

.. warning::

    This step will delete everything in the repository that isn't already
    tracked by git. This is not reversible. Be careful!

To clean up the repo (double warning: this deletes everything not already
tracked by git), run::

    git clean -dfx  # warning: this deletes everything not tracked by git

Next we use PEP517 to build the source distribution::

    pip install pep517
    python -m pep517.build --source --out-dir dist .

There should now be a ``.tar.gz`` file in the ``dist`` directory which
contains the package as it will appear on PyPI. Unpack it, and check that it
looks the way you expect it to.

To validate the package and upload to PyPI, run::

    pip install twine
    twine check dist/*
    twine upload dist/spotdot*.tar.gz

For more on package releases, check out `the OpenAstronomy packaging guide
<https://packaging-guide.openastronomy.org/en/latest/releasing.html>`_.

