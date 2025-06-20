"""
Plotting utilities for corpus analysis visualization.

This package provides plotting and charting functions for various
types of corpus analysis data visualization.
"""

# Import interactive plotting functions with highlighting and grouping
from webapp.utilities.plotting.interactive_plots import (
    plot_grouped_boxplot,
    plot_pca_scatter_highlight,
    plot_pca_variable_contrib_bar,
    plot_scatter,
    plot_scatter_highlight
)

from webapp.utilities.plotting.advanced_plots import (
    generate_boxplot,
    generate_boxplot_by_group
)
from webapp.utilities.plotting.advanced_analysis import (
    generate_pca,
    generate_scatterplot,
    generate_scatterplot_with_groups,
    update_pca_plot
)
from webapp.utilities.plotting.boxplot_utils import (
    boxplots_pl
)

from webapp.utilities.plotting.charts import (
    plot_tag_frequencies_bar,
    plot_compare_corpus_bar,
    plot_general_boxplot,
    plot_download_link,
    clear_scatterplot_multiselect,
    clear_boxplot_multiselect,
    clear_plots,
    update_pca_idx_tab1,
    update_pca_idx_tab2,
    update_grpa,
    update_grpb,
    update_tar,
    update_ref
)

__all__ = [
    'plot_tag_frequencies_bar',
    'plot_compare_corpus_bar',
    'plot_general_boxplot',
    'plot_download_link',
    'clear_scatterplot_multiselect',
    'clear_boxplot_multiselect',
    'clear_plots',
    'update_pca_idx_tab1',
    'update_pca_idx_tab2',
    'update_grpa',
    'update_grpb',
    'update_tar',
    'update_ref',
    'plot_grouped_boxplot',
    'plot_pca_scatter_highlight',
    'plot_pca_variable_contrib_bar',
    'plot_scatter',
    'plot_scatter_highlight',
    'plot_tag_density',
    'generate_boxplot',
    'generate_boxplot_by_group',
    'generate_pca',
    'generate_scatterplot',
    'generate_scatterplot_with_groups',
    'update_pca_plot',
    'boxplots_pl'
]
