{{objname}}
{{ underline }}==============

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:
   :private-members:

   {% block methods %}
   .. automethod:: __init__
   {% endblock %}

.. include:: /modules/generated/backreferences/{{module}}.{{objname}}.examples
