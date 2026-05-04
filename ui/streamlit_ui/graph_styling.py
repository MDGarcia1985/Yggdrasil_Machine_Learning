"""
Graph / Canvas configuration.

Copyright 2026 M&E Design
Created by
Michael Garcia - michael@mandedesign.studio
Aaron Hurst - https://github.com/hurstaaron
Joseph Haskins - https://github.com/discreet6247
"""
# ----------------------------------------
# Graph / Canvas configuration
# ----------------------------------------

def get_graph_config(view: str = "Concept"):
    """
    Returns rendering configuration for the circuit graph UI.
    """

    node_size = 70 if view == "Technical" else 52

    return {
        "canvas_width": 1200,
        "canvas_height": 700,
        "node_size": node_size,

        # Dark UI works well for circuit graphs
        "background": "#2B2B2B",

        # Subtle grid (not dominant like TinyTrainer)
        "grid_color": "#444444",
    }
