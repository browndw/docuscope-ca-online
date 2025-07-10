"""
Plotting utilities for corpus analysis visualization.

This package provides plotting and charting functions for various
types of corpus analysis data visualization.
"""

# Import interactive plotting functions with highlighting and grouping
from webapp.utilities.plotting.bar_charts import (
    plot_tag_frequencies_bar,
    plot_compare_corpus_bar,
    plot_pca_variable_contrib_bar,
    plot_tag_density
)

from webapp.utilities.plotting.boxplots import (
    plot_general_boxplot,
    plot_grouped_boxplot,
    generate_boxplot,
    generate_boxplot_by_group
)
from webapp.utilities.plotting.scatterplots import (
    plot_scatter,
    plot_scatter_highlight,
    generate_scatterplot,
    generate_scatterplot_with_groups
)
from webapp.utilities.plotting.pca_plots import (
    plot_pca_scatter_highlight,
    generate_pca,
    update_pca_plot
)
from webapp.utilities.plotting.utils import (
    plot_download_link,
    clear_scatterplot_multiselect,
    clear_boxplot_multiselect,
    clear_plot_toggle,
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
    'clear_plot_toggle',
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
