"""
Fig. 4 of the paper: how much attention the label token pays to each feature.

For a serialized row we take the attention from the target-VALUE token (averaged over heads, last
`n_layers` layers) to every earlier token and sum it per field. With the target at the end
(permute_x) every feature is visible; with the target shuffled to the front (permute_xy) the label
token cannot attend to features that come after it, which is the paper's argument for permute_x.
"""

from typing import Any, Optional, List, Dict, Union

def field_attention(model, tok, text, fields, target_field, n_layers):
    pass

def plot_attention_comparison(results, path, title):
    """results: {'permute_x (target last)': {field: score}, 'permute_xy (target first)': {...}}"""
    pass
