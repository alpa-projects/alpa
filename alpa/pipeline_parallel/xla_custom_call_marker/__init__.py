try:
    from .build.xla_custom_call_marker import pipeline_marker, identity
except ImportError as e:
    import os
    print("Cannot import XLA custom markers")
    path = os.path.dirname(__file__)
    print(f"Please run 'bash build.sh' under {path}")
    exit(-1)
