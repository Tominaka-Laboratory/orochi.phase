.. image:: images/orochi_icon.png
   :align: center
   :width: 80px
   :target: https://github.com/Tominaka-Laboratory/orochi.phase
   :alt: Orochi Logo

============
orochi.phase
============

.. |title| replace:: orochi.phase
.. _title: https://Tominaka-Laboratory.github.io/orochi.phase

.. |Icon| image:: images/orochi_icon.png
   :target: https://github.com/Tominaka-Laboratory/orochi.phase
   :height: 100px

|PyPi| |Forge| |PythonVersion| |ChemRxiv| |PR|

|CI| |Codecov| |Black| |Tracking|

.. |ChemRxiv| image:: https://img.shields.io/badge/ChemRxiv-10.26434%2Fchemrxiv.15010802-blue.svg
        :target: https://doi.org/10.26434/chemrxiv.15010802

.. |Black| image:: https://img.shields.io/badge/code_style-black-black
        :target: https://github.com/psf/black

.. |CI| image:: https://github.com/Tominaka-Laboratory/orochi.phase/actions/workflows/tests.yml/badge.svg
        :target: https://github.com/Tominaka-Laboratory/orochi.phase/actions/workflows/tests.yml

.. |Codecov| image:: https://codecov.io/gh/Tominaka-Laboratory/orochi.phase/branch/main/graph/badge.svg
        :target: https://codecov.io/gh/Tominaka-Laboratory/orochi.phase

.. |Forge| image:: https://img.shields.io/conda/vn/conda-forge/orochi.phase
        :target: https://anaconda.org/conda-forge/orochi.phase

.. |PR| image:: https://img.shields.io/badge/PR-Welcome-29ab47ff

.. |PyPi| image:: https://img.shields.io/pypi/v/orochi.phase
        :target: https://pypi.org/project/orochi.phase/

.. |PythonVersion| image:: https://img.shields.io/pypi/pyversions/orochi.phase
        :target: https://pypi.org/project/orochi.phase/

.. |Tracking| image:: https://img.shields.io/badge/issue_tracking-github-blue
        :target: https://github.com/Tominaka-Laboratory/orochi.phase/issues


Overview
========

The ``orochi.phase`` package is a versatile Python module for managing and manipulating
the structural information of diverse materials. It provides a comprehensive
toolkit for working with atomic structures, ranging from three-dimensionally
periodic crystals to non-periodic systems and finite-sized nanostructures.
As a core component of the Orochi framework, it facilitates detailed structural
analysis and model construction.

Key Features
============

* **Comprehensive Structure Representation**:
    * Handles a wide array of material structures, including 3D bulk crystals,
        2D materials (e.g., nanosheets), nanoparticles, and other aperiodic or
        finite systems.

* **Advanced Crystallographic Tools**:
    * Enables the description of crystal structures starting from the asymmetric unit.
    * Automatically generates full atomic coordinates within the unit cell based on
        space group symmetry operations.
    * Allows for the application of constraints on atomic coordinates derived from
        site symmetry information.

* **Customizable Periodic Boundary Conditions (PBCs)**:
    * Offers flexibility in defining periodicity, allowing PBCs to be restricted
        along arbitrary directions. This is particularly useful for modeling
        systems with reduced dimensionality, such as surfaces or 2D materials.

* **Morphology and Nanostructure Modeling**:
    * Facilitates the explicit generation of atomic models for materials exhibiting
        specific morphologies (e.g., nanoparticles, thin films, surfaces).
        These models can be derived from bulk crystalline data, enabling the study
        of finite-size and surface effects.

Quickstart: Structure Model Generation & Anisotropic PBCs
==========================================================

.. code-block:: python

    from orochi.phase import Phase

    # 1. Initialize Phase and load structure
    phase = Phase()
    phase.read_file("PtO2.cif")

    # 2. Toggle Anisotropic Periodic Boundary Conditions
    # Seamlessly model 2D monolayers, 1D wires, or 3D bulk without building artificial supercells:
    phase["pbc"] = [True, True, False]   # 2D monolayer (ab-periodic, non-periodic c)

    # 3. Choose morphology representation
    # 'analytical' for bulk crystals / analytical spherical envelopes
    # 'sphere' / 'cube' for explicit coordination-preserving nanoparticles
    phase["shape_type"] = "analytical"

    # 4. Enforce space-group Wyckoff symmetry and lattice constraints
    phase.set_lattice_constraints()
    
    # 5. Execute structural expansion
    phase.run(force=True)

    # 6. Access expanded fractional coordinates and atomic species
    xyz = phase["xyz"]
    elements = phase["element"]
    print(f"Total atoms in expanded cluster/cell: {len(elements)}")

Citation
--------

If you use orochi.phase in a scientific publication, please cite this package as:

    S. Tominaka, *Orochi: A Physics-Grounded Python Package for Pair Distribution Function Analysis of Crystalline and Low-Dimensional Materials*, ChemRxiv (2026). DOI: 10.26434/chemrxiv.15010802
    https://doi.org/10.26434/chemrxiv.15010802

Installation
------------

The preferred method is to install using ``pip``:

.. code-block:: bash

    pip install orochi.phase

To confirm that the installation was successful, type:

.. code-block:: bash

    python -c "import orochi.phase; print('Orochi Phase imported successfully')"

If you prefer to install from source:

.. code-block:: bash

    git clone https://github.com/Tominaka-Laboratory/orochi.phase.git
    cd orochi.phase
    pip install -e .

Getting Started
---------------

You may consult our `online documentation <https://Tominaka-Laboratory.github.io/orochi.phase>`_ for tutorials and API references.

Support and Contribute
----------------------

The orochi.phase project welcomes your expertise and enthusiasm!

If you see a bug or want to request a feature, please `report it as an issue <https://github.com/Tominaka-Laboratory/orochi.phase/issues>`_ and/or `submit a fix as a PR <https://github.com/Tominaka-Laboratory/orochi.phase/pulls>`_.

Feel free to fork the project and contribute. To install orochi.phase
in a development mode, with its sources being directly used by Python
rather than copied to a package directory, use the following in the root
directory ::

        pip install -e .

To ensure code quality and to prevent accidental commits into the default branch, please set up the use of our pre-commit
hooks.

Improvements and fixes are always appreciated.

Before contributing, please read our `Code of Conduct <https://github.com/Tominaka-Laboratory/orochi.phase/blob/main/CODE_OF_CONDUCT.rst>`_.



License
-------

This project is licensed under the **GNU General Public License v3.0 (GPLv3)**. See the `LICENSE.rst` file for details.

This copyleft license guarantees that Orochi remains freely accessible, open, and transparent to the entire scientific community, and that all derivative works must likewise remain open source.

Citation
--------

If you use this package in academic research, please cite:

.. code-block:: bibtex

    @article{tominaka2026orochi,
      title   = {Orochi: A Physics-Grounded Python Package for Pair Distribution Function Analysis of Crystalline and Low-Dimensional Materials},
      author  = {Tominaka, Satoshi},
      journal = {ChemRxiv},
      year    = {2026},
      doi     = {10.26434/chemrxiv.15010802},
      url     = {https://doi.org/10.26434/chemrxiv.15010802}
    }

Contact
-------

For more information on orochi.phase please visit the project `web-page <https://Tominaka-Laboratory.github.io/>`_ or email Dr. Satoshi Tominaka at tominaka.satoshi@nims.go.jp.
