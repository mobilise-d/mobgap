{{objname}}
{{ underline }}==============

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :exclude-members: PredefinedParameters, HyperparameterSearchPresets

   {% block methods %}
   .. automethod:: __init__
   {% endblock %}

.. include:: /modules/generated/backreferences/{{module}}.{{objname}}.examples
