=====
yeoda
=====

*yeoda* stands for **y**\ our **e**\ arth **o**\ bservation **d**\ ata **a**\ ccess and provides lower and higher-level data cube
classes to work with well-defined and structured earth observation data. These data cubes allow to filter, split and load data independently from the way the data is structured on the hard disk. Once the data structure is known to *yeoda*, it offers a user-friendly interface to access the data with the aforementioned operations.
Internally, the package relies on functionalities provided by `geopathfinder <https://github.com/TUW-GEO/geopathfinder>`_
(filepath/filename and folder structure handling library), `veranda <https://github.com/TUW-GEO/veranda>`_ (IO classes and higher-level data structure classes for vector and raster data)
and `geospade <https://github.com/TUW-GEO/geospade>`_ (raster and vector geometry definitions and operations).
Moreover, another very important part of *yeoda* is work with pre-defined grids like the `Equi7Grid <https://github.com/TUW-GEO/Equi7Grid>`_ or the `LatLonGrid <https://github.com/TUW-GEO/latlongrid>`_.
These grid packages can simplify and speed up spatial operations to identify tiles/files of interest (e.g, bounding box request by a user).


Contents
========

.. toctree::
   :maxdepth: 2

   Examples <examples>
   Installation <install>
   Module Reference <api/modules>
   License <license>
   Authors <authors>
   Changelog <changelog>



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _toctree: http://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html
.. _reStructuredText: http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
.. _references: http://www.sphinx-doc.org/en/stable/markup/inline.html
.. _Python domain syntax: http://sphinx-doc.org/domains.html#the-python-domain
.. _Sphinx: http://www.sphinx-doc.org/
.. _Python: http://docs.python.org/
.. _Numpy: http://docs.scipy.org/doc/numpy
.. _SciPy: http://docs.scipy.org/doc/scipy/reference/
.. _matplotlib: https://matplotlib.org/contents.html#
.. _Pandas: http://pandas.pydata.org/pandas-docs/stable
.. _Scikit-Learn: http://scikit-learn.org/stable
.. _autodoc: http://www.sphinx-doc.org/en/stable/ext/autodoc.html
.. _Google style: https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings
.. _NumPy style: https://numpydoc.readthedocs.io/en/latest/format.html
.. _classical style: http://www.sphinx-doc.org/en/stable/domains.html#info-field-lists
