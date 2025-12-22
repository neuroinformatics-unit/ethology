{{ name | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:

   {% block methods %}
   {% set ns = namespace(has_public_methods=false) %}

   {% if methods %}
   {% for item in methods %}
   {% if not item.startswith('_') %}
   {% set ns.has_public_methods = true %}
   {% endif %}
   {%- endfor %}
   {% endif %}

   {% if ns.has_public_methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
   {% for item in methods %}
   {% if not item.startswith('_') %}
      {{ item|is_own_method(name, module) }}
   {% endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   .. rubric:: {{ _('Details') }}

.. minigallery:: {{ module }}.{{ objname }}
   :add-heading: Examples using ``{{ objname }}``
