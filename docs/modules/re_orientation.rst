Reorientation
=============
.. automodule:: mobgap.re_orientation
    :no-members:
    :no-inherited-members:

Algorithms
++++++++++
.. currentmodule:: mobgap.re_orientation
.. autosummary::
   :toctree: generated/re_orientation
   :template: class.rst

    ReorientationMethodDM

Pipelines
+++++++++
.. currentmodule:: mobgap.re_orientation

.. autosummary::
   :toctree: generated/re_orientation
   :template: class.rst

    pipeline.ReorientationEmulationPipeline

Evaluation Scores
+++++++++++++++++
These scores are expected to be used in combination with :class:`~mobgap.utils.evaluation.Evaluation` and
:class:`~mobgap.utils.evaluation.EvaluationCV` or directly with :func:`~tpcp.validation.cross_validation` and
:func:`~tpcp.validation.validation`.

.. currentmodule:: mobgap.re_orientation.evaluation

.. autosummary::
   :toctree: generated/re_orientation

    reorientation_score

.. autosummary::
   :toctree: generated/re_orientation
   :template: func.rst

    reorientation_per_datapoint_score
    reorientation_final_agg

Base Classes
++++++++++++
.. automodule:: mobgap.re_orientation.base
    :no-members:
    :no-inherited-members:

.. currentmodule:: mobgap.re_orientation.base

.. autosummary::
   :toctree: generated/re_orientation
   :template: class.rst

    BaseReorientationCorrector
