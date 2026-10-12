{{objname}}
{{ underline }}==============

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :exclude-members: PredefinedParameters, OptimizationPresets

   {% block methods %}
   .. automethod:: __init__
   {% endblock %}

.. include:: /modules/generated/backreferences/{{module}}.{{objname}}.examples
