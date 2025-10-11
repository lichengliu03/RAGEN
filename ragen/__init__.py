"""RAGEN package initialisation."""

from ragen.patches import apply_omega_conf_patch

# Ensure VERL config instantiation accepts RAGEN-specific extensions.
apply_omega_conf_patch()
