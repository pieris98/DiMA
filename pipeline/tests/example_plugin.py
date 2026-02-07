"""
Example third-party plugin for testing.

This demonstrates how an external party would create a plugin for DiMA.
"""


class CustomSequenceLengthMetric:
    """Example custom metric: average sequence length."""
    name = "avg_seq_length"
    requires_references = False
    is_per_sample = False

    def compute(self, predictions, references=None, **kwargs):
        if not predictions:
            return 0.0
        return sum(len(s) for s in predictions) / len(predictions)


class CustomStage:
    """Example custom pipeline stage."""
    name = "custom_hello"
    description = "A test stage that just prints hello"

    def validate(self, config, context):
        return []

    def run(self, config, context):
        context["custom_hello_ran"] = True
        print("[custom_hello] Hello from the plugin!")
        return context


def register(registry):
    """Register all components from this plugin."""
    registry.register("metric", "avg_seq_length", CustomSequenceLengthMetric)
    registry.register("stage", "custom_hello", CustomStage)
