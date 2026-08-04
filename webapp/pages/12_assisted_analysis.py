"""
This app provides an interface for deterministic DFM statistics.
"""

import streamlit as st
import docuscospacy as ds

# Module-specific imports
from webapp.utilities.session import (
    get_or_init_user_session, safe_session_get, load_metadata
)
from webapp.utilities.analysis import (
    available_dfm_features,
    compute_dfm_statistics,
    generate_tags_table,
    get_dfm_statistic_options,
)
from webapp.utilities.corpus import get_corpus_data_manager
from webapp.utilities.ui import (
    sidebar_help_link, render_table_generation_interface
)
from webapp.utilities.state import (
    SessionKeys, WarningKeys, CorpusKeys
)
from webapp.menu import (
    menu, require_login
)

TITLE = "Matrix Explorer"
ICON = ":material/table_view:"

st.set_page_config(
    page_title=TITLE, page_icon=ICON,
    layout="wide"
)

DFM_SOURCES = {
    "DocuScope DFM": {
        "key": "dtm_ds",
        "tagset": "DocuScope",
        "transform": None,
    },
    "Parts-of-Speech DFM": {
        "key": "dtm_pos",
        "tagset": "Parts-of-Speech",
        "transform": None,
    },
    "Parts-of-Speech DFM (general)": {
        "key": "dtm_pos",
        "tagset": "Parts-of-Speech",
        "transform": ds.dtm_simplify,
    },
}


def _metadata_categories(metadata_target: dict | None) -> list[str]:
    """Return filename-derived metadata categories when available."""
    if not metadata_target:
        return []
    doccats = metadata_target.get("doccats", [])
    if isinstance(doccats, list) and doccats:
        first = doccats[0]
        if isinstance(first, dict):
            return list(first.get("cats", []))
    if isinstance(doccats, dict):
        return list(doccats.get("cats", []))
    return []


def _load_selected_dfm(user_session_id: str, source_label: str):
    """Load one target DFM source from the corpus data manager."""
    source = DFM_SOURCES[source_label]
    manager = get_corpus_data_manager(user_session_id, CorpusKeys.TARGET)
    dfm = manager.get_data(source["key"])
    if dfm is None:
        return None, source
    transform = source.get("transform")
    if transform is not None:
        dfm = transform(dfm)
    return dfm, source


def _build_group_selection(
    metadata_groups: list[str],
    group_mode: str,
    selected_groups: list[str] | None,
    group_a: list[str] | None,
    group_b: list[str] | None,
) -> tuple[list[str] | None, list[str] | None, str | None]:
    """Build row-aligned group labels and optional group filters from UI controls."""
    if group_mode == "No grouping":
        return None, None, None
    if group_mode == "Metadata groups":
        return metadata_groups, selected_groups or None, None

    group_a = group_a or []
    group_b = group_b or []
    overlap = sorted(set(group_a).intersection(group_b))
    if overlap:
        overlap_labels = ", ".join(overlap)
        return None, None, f"Categories cannot be in both group A and group B: {overlap_labels}"
    if not group_a or not group_b:
        return None, None, "Select at least one category for group A and group B."

    recoded_groups = [
        "Group A" if group in group_a else "Group B" if group in group_b else "Other"
        for group in metadata_groups
    ]
    return recoded_groups, ["Group A", "Group B"], None


def _render_statistics_table(table, statistic: str) -> None:
    """Render a statistics table with the selected statistic visually emphasized."""
    if statistic not in table.columns:
        st.dataframe(table, width='stretch', hide_index=True)
        return

    styled_table = table.to_pandas().style.set_properties(
        subset=[statistic],
        **{"background-color": "#fff3bf", "font-weight": "700"},
    )
    st.dataframe(styled_table, width='stretch', hide_index=True)


def render_dfm_explorer(user_session_id: str, session: dict) -> None:
    """Render deterministic DFM statistics controls and result table."""
    metadata_target = None
    if safe_session_get(session, SessionKeys.HAS_TARGET, False):
        metadata_target = load_metadata(CorpusKeys.TARGET, user_session_id)
    metadata_groups = _metadata_categories(metadata_target)

    st.markdown(
        body=(
            ":material/table_view: Select a document-feature matrix, choose a statistic, "
            "and return a computed table. No model call is needed for this step."
        )
    )

    source_label = st.selectbox(
        "DFM source",
        options=list(DFM_SOURCES.keys()),
        key="dfm_explorer_source",
    )
    dfm, source = _load_selected_dfm(user_session_id, source_label)
    if dfm is None:
        st.warning("The selected DFM is not available yet.", icon=":material/info:")
        return

    selectable_features = available_dfm_features(dfm, source["tagset"])
    feature_scope = st.segmented_control(
        "Feature scope",
        ["All features", "Selected features"],
        default="All features",
        key="dfm_explorer_feature_scope",
        help="Use all available features or narrow the statistic to selected tags.",
    )
    selected_features = None
    if feature_scope == "Selected features":
        selected_features = st.segmented_control(
            "Features",
            selectable_features,
            selection_mode="multi",
            key="dfm_explorer_selected_features",
            help="Choose one or more DFM features to include in the computed table.",
        )
        if not selected_features:
            st.warning("Select at least one feature.", icon=":material/info:")
            return

    group_choices = ["No grouping"]
    if metadata_groups:
        group_choices.extend(["Metadata groups", "Compare group sets"])
    group_mode = st.segmented_control(
        "Grouping",
        group_choices,
        default="No grouping",
        key="dfm_explorer_grouping",
        help="Use metadata categories as groups, or recode categories into group A and B.",
    )
    selected_groups = None
    group_a = None
    group_b = None
    all_groups = sorted(set(metadata_groups))
    if group_mode == "Metadata groups":
        selected_groups = st.segmented_control(
            "Groups",
            all_groups,
            selection_mode="multi",
            key="dfm_explorer_selected_groups",
            help="Choose which metadata groups to include.",
        )
        if not selected_groups:
            st.warning("Select at least one group.", icon=":material/info:")
            return
    elif group_mode == "Compare group sets":
        group_a = st.segmented_control(
            "Select categories for group A",
            all_groups,
            selection_mode="multi",
            key="dfm_explorer_group_a",
            help="Group A can combine multiple metadata categories.",
        )
        group_b = st.segmented_control(
            "Select categories for group B",
            all_groups,
            selection_mode="multi",
            key="dfm_explorer_group_b",
            help="Group B can combine multiple metadata categories.",
        )

    statistic_options = get_dfm_statistic_options()
    statistic = st.segmented_control(
        "Statistic",
        list(statistic_options.keys()),
        default="mean_rf",
        format_func=lambda key: statistic_options[key],
        key="dfm_explorer_statistic",
    )
    rank_axis_options = ["features"]
    if group_mode != "No grouping":
        rank_axis_options.append("groups")
    rank_axis = st.segmented_control(
        "Rank",
        rank_axis_options,
        default="features",
        format_func=lambda value: "Features" if value == "features" else "Groups",
        key="dfm_explorer_rank_axis",
    )
    rank_order = st.segmented_control(
        "Order",
        ["Highest", "Lowest"],
        default="Highest",
        key="dfm_explorer_rank_order",
    )
    limit = st.number_input(
        "Rows to return",
        min_value=1,
        max_value=100,
        value=10,
        step=1,
        key="dfm_explorer_limit",
    )

    groups, selected_groups_filter, group_error = _build_group_selection(
        metadata_groups=metadata_groups,
        group_mode=group_mode,
        selected_groups=selected_groups,
        group_a=group_a,
        group_b=group_b,
    )
    if group_error:
        st.warning(group_error, icon=":material/info:")
        return
    if groups is not None and len(groups) != dfm.height:
        st.warning(
            "Metadata categories are not aligned with the selected DFM rows.",
            icon=":material/info:",
        )
        groups = None

    try:
        result = compute_dfm_statistics(
            dfm=dfm,
            statistic=statistic,
            tagset=source["tagset"],
            rank_axis=rank_axis,
            groups=groups,
            selected_features=selected_features,
            selected_groups=selected_groups_filter,
            descending=rank_order == "Highest",
            limit=int(limit),
        )
    except Exception as exc:
        st.error(f"Could not compute DFM statistics: {exc}", icon=":material/error:")
        return

    st.info(result.note, icon=":material/info:")
    _render_statistics_table(result.table, statistic)

    csv = result.table.write_csv()
    st.download_button(
        "Download table",
        data=csv,
        file_name="dfm_statistics.csv",
        mime="text/csv",
        icon=":material/file_download:",
    )


def main():
    """Main function to run the Streamlit app for DFM exploration."""
    # Set login requirements for navigation
    require_login()
    menu()
    st.markdown(
        body=f"## {TITLE}",
        help=(
            "Select a document-feature matrix and a statistic to return a computed "
            "analysis table. This deterministic workflow does not require an AI model."
        )
    )

    # Get or initialize user session
    user_session_id, session = get_or_init_user_session()

    # Add help link
    sidebar_help_link("assisted-analysis.html")

    # Check if tags table is available
    if safe_session_get(session, SessionKeys.TAGS_TABLE, False):
        render_dfm_explorer(user_session_id, session)
    else:
        # Show generation interface for tags table
        render_table_generation_interface(
            user_session_id=user_session_id,
            session=session,
            table_type="tags table",
            button_label="Load Tables",
            generation_func=generate_tags_table,
            session_key=SessionKeys.TAGS_TABLE,
            warning_key=WarningKeys.TAGS
        )

    st.sidebar.markdown("---")


if __name__ == "__main__":
    main()
