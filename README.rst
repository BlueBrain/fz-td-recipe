fz-td-recipe
============

A wrapper around a connectome building recipe used by BlueBrain tools, to be either
defined in JSON/YAML or XML (deprecated).

Installation
------------

To install the latest stable release, please use:

.. code-block:: console

   $ pip install fz-td-recipe

Or to get the latest code, build it from ``git``:

.. code-block:: console

   $ gh repo clone BlueBrain/fz-td-recipe
   $ cd fz-td-recipe
   $ pip install .

Usage
-----

A small command line utility is provided to convert legacy XML recipes and validate
whether a JSON or YAML file conforms to the recipe schema:

.. code-block:: console

   $ fz-td-recipe convert recipe.xml recipe.json
   $ fz-td-recipe validate recipe.json

More details and features can be accessed with i.e.:

.. code-block:: console

   $ fz-td-recipe validate --help

Further Documentation
---------------------

The recipe format is documented within the `SONATA extension`_ by the Blue Brain Project.

Acknowledgment
--------------
The development of this software was supported by funding to the Blue Brain Project,
a research center of the École polytechnique fédérale de Lausanne (EPFL),
from the Swiss government's ETH Board of the Swiss Federal Institutes of Technology.

Copyright (c) 2021-2024 Blue Brain Project/EPFL

.. _SONATA extension: https://sonata-extension.readthedocs.io/en/latest/recipe.html
