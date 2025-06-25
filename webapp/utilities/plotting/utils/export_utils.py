"""
Boxplot utilities for corpus analysis visualization.

This module provides functions for preparing data and generating
boxplots for corpus analysis.
"""
import base64
import streamlit as st
import plotly.graph_objects as go


def plot_download_link(
        fig: go.Figure,
        filename="plot.png",
        scale=2,
        button_text="Download high-res PNG"
        ) -> None:
    """
    Display a download link for a high-resolution PNG of a Plotly figure.

    Parameters
    ----------
    fig : plotly.graph_objs.Figure
        The Plotly figure to export.
    filename : str
        The filename for the downloaded PNG.
    scale : int or float
        The scale factor for the image resolution (default 2).
    button_text : str
        The text to display for the download link/button.

    Returns
    -------
    None
        Renders a download link in the Streamlit app.
    """
    # Export the figure to a PNG byte stream
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    img_bytes = fig.to_image(format="png", scale=scale)
    b64 = base64.b64encode(img_bytes).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">{button_text}</a>'
    st.markdown(href, unsafe_allow_html=True)
