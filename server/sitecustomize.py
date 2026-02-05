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

try:
    from transformers.modeling_utils import PreTrainedModel

    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        # transformers>=5 expects all_tied_weights_keys on PreTrainedModel.
        # Provide a backward-compatible property for custom models.
        @property
        def all_tied_weights_keys(self):
            keys = getattr(self, "_tied_weights_keys", None)
            if keys is None:
                return {}
            if isinstance(keys, dict):
                return keys
            if isinstance(keys, (list, tuple, set)):
                return {k: None for k in keys}
            return {}

        PreTrainedModel.all_tied_weights_keys = all_tied_weights_keys
except Exception:
    # If transformers isn't installed, do nothing.
    pass
