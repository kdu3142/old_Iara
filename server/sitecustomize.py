"""
Runtime compatibility shims loaded automatically by Python.

This patches phonemizer's EspeakWrapper for versions that don't expose
set_data_path (required by misaki/espeak).
"""

try:
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    if not hasattr(EspeakWrapper, "set_data_path"):
        # Newer phonemizer exposes data_path property but not setter.
        # Provide a compatible classmethod expected by misaki.
        @classmethod
        def set_data_path(cls, path):
            cls.data_path = path

        EspeakWrapper.set_data_path = set_data_path
except Exception:
    # If phonemizer isn't installed, do nothing.
    pass
