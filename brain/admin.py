"""Admin for users app."""
from django.contrib import admin

# pylint: disable=no-name-in-module
from brain.models import (BrainNetwork, BrainPattern, BrainRetrieval)


@admin.register(BrainNetwork, BrainPattern, BrainRetrieval)
class UniversalAdmin(admin.ModelAdmin):
    """Universal admin for all models."""
    def get_list_display(self, request):
        """Return list display."""
        # pylint: disable=protected-access
        return [field.name for field in self.model._meta.concrete_fields]
