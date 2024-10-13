from typing import List

import plotly.graph_objects as go
import plotly.io as pio
from evidently.options import ColorOptions
from pydantic import BaseModel


class FSDSColors(BaseModel):
    main: str = "#c60000"
    others: List[str] = [
        "#0011C7",
        "#00C738",
        "#C79C00",
        "#722626",
    ]  # Adobe Color Wheel > Square


fsds_colors = FSDSColors()

color_scheme = ColorOptions(
    primary_color=fsds_colors.main,
    fill_color="#fff4f2",
    zero_line_color="#016795",
    current_data_color=fsds_colors.main,
    reference_data_color=fsds_colors.others[0],
)

fsds_template = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Inter Variable, Inter"),
        title={
            "font": {
                "family": "Inter Variable, Inter Tight, Inter",
                "weight": "bold",
            }
        },
        colorway=[fsds_colors.main, *fsds_colors.others],
        template="simple_white",
        xaxis=dict(
            showgrid=False
        ),  # Do not put tickmode='linear' hear as it can cause Plotly to mistread xaxis as integer instead of category
        yaxis=dict(showgrid=False, showticklabels=False, tickformat=","),
        showlegend=False,
    )
)

pio.templates["fsds"] = fsds_template
pio.templates.default = "fsds"
