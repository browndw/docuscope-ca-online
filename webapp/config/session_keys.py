# Copyright (C) 2025 David West Brown

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Functions for generating content.
# Some populate descriptive content from corpora or sub-corpora.
# Others populate the results of statistical functions.

class SessionKeys:
    # Session DataFrame container
    SESSION_DATAFRAME = "session"
    # Session state flags stored in the DataFrame
    HAS_TARGET = "has_target"
    TARGET_DB = "target_db"
    HAS_META = "has_meta"
    HAS_REFERENCE = "has_reference"
    REFERENCE_DB = "reference_db"
    FREQ_TABLE = "freq_table"
    TAGS_TABLE = "tags_table"
    KEYNESS_TABLE = "keyness_table"
    NGRAMS = "ngrams"
    KWIC = "kwic"
    KEYNESS_PARTS = "keyness_parts"
    DTM = "dtm"
    PCA = "pca"
    COLLOCATIONS = "collocations"
    DOC = "doc"
    METADATA_TARGET = "metadata_target"
    METADATA_REFERENCE = "metadata_reference"

    # AI Assistant State Keys
    AI_PLOTBOT_CHAT = "plotbot"
    AI_PANDABOT_CHAT = "pandasai"
    AI_USER_KEY = "user_key"
    AI_PLOT_INTENT = "plot_intent"

    # AI Counter Keys
    AI_PLOTBOT_PROMPT_COUNT = "plotbot_user_prompt_count"
    AI_PANDABOT_PROMPT_COUNT = "pandabot_user_prompt_count"

    # AI Cache Keys
    AI_PLOTBOT_CACHE = "plotbot_cache"

    # AI Widget Persistence Keys
    AI_PLOTBOT_QUERY = "plotbot_query"
    AI_PLOTBOT_CORPUS = "plotbot_corpus"
    AI_PLOTBOT_PERSIST = "assisted_plotting_PERSIST"
    AI_PLOTBOT_PIVOT_TABLE = "pivot_table"
    AI_PLOTBOT_MAKE_PERCENT = "make_percent"


class MetadataKeys:
    TOKENS_POS = "tokens_pos"
    TOKENS_DS = "tokens_ds"
    NDOCS = "ndocs"
    MODEL = "model"
    DOCIDS = "docids"
    TAGS_DS = "tags_ds"
    TAGS_POS = "tags_pos"
    DOCCATS = "doccats"
    COLLOCATIONS = "collocations"
    KEYNESS_PARTS = "keyness_parts"
    VARIANCE = "variance"


class CorpusKeys:
    TARGET = "target"
    REFERENCE = "reference"


class TargetKeys:
    DS_TOKENS = "ds_tokens"
    FT_POS = "ft_pos"
    FT_DS = "ft_ds"
    TT_POS = "tt_pos"
    TT_DS = "tt_ds"
    DTM_POS = "dtm_pos"
    DTM_DS = "dtm_ds"
    KW_POS = "kw_pos"
    KW_DS = "kw_ds"
    KT_POS = "kt_pos"
    KT_DS = "kt_ds"
    KW_POS_CP = "kw_pos_cp"
    KW_DS_CP = "kw_ds_cp"
    KT_POS_CP = "kt_pos_cp"
    KT_DS_CP = "kt_ds_cp"
    NGRAMS = "ngrams"
    KWIC = "kwic"
    COLLOCATIONS = "collocations"
    DOC_POS = "doc_pos"
    DOC_SIMPLE = "doc_simple"
    DOC_DS = "doc_ds"
    PCA_DF = "pca_df"
    CONTRIB_DF = "contrib_df"


class ReferenceKeys:
    DS_TOKENS = "ds_tokens"
    FT_POS = "ft_pos"
    FT_DS = "ft_ds"
    TT_POS = "tt_pos"
    TT_DS = "tt_ds"


class WarningKeys:
    LOAD_CORPUS = "warning"
    FREQUENCY = "frequency_warning"
    TAGS = "tags_warning"
    KEYNESS = "keyness_warning"
    KEYNESS_PARTS = "keyness_parts_warning"
    COLLOCATIONS = "collocations_warning"
    KWIC = "kwic_warning"
    NGRAM = "ngram_warning"
    DOC = "doc_warning"
    BOX = "boxplot_warning"
    BOX_GROUP = "boxplot_group_warning"
    SCATTER = "scatter_warning"
    SCATTER_GROUP = "scatter_group_warning"
    PCA = "pca_warning"


class LoadCorpusKeys:
    READY_TO_PROCESS = "ready_to_process"
    EXCEPTIONS = "exceptions"
    REF_EXCEPTIONS = "ref_exceptions"
    MODEL = "model"
    CORPUS_DF = "corpus_df"
    REF_CORPUS_DF = "ref_corpus_df"
    REF_READY_TO_PROCESS = "ref_ready_to_process"


class BoxplotKeys:
    ATTEMPTED = "boxplot_attempted"
    GROUP_ATTEMPTED = "boxplot_group_attempted"
    DF = "boxplot_df"
    STATS = "boxplot_stats"
    WARNING = "boxplot_warning"
    GROUP_DF = "boxplot_group_df"
    GROUP_STATS = "boxplot_group_stats"
    GROUP_WARNING = "boxplot_group_warning"
    CONFIRMED_VAL1 = "confirmed_box_val1"
    CONFIRMED_VAL2 = "confirmed_box_val2"
    CONFIRMED_GRPA = "confirmed_grpa"
    CONFIRMED_GRPB = "confirmed_grpb"


class ScatterplotKeys:
    ATTEMPTED = "scatterplot_attempted"
    GROUP_ATTEMPTED = "scatterplot_group_attempted"
    DF = "scatterplot_df"
    GROUP_DF = "scatterplot_group_df"
    WARNING = "scatter_warning"
    GROUP_WARNING = "scatter_group_warning"
    CORRELATION = "scatter_correlation"
    GROUP_CORRELATION = "scatter_group_correlation"
    GROUP_X = "scatterplot_group_x"
    GROUP_Y = "scatterplot_group_y"
    GROUP_SELECTED_GROUPS = "scatterplot_group_selected_groups"


class PCAKeys:
    ATTEMPTED = "pca_attempted"
    WARNING = "pca_warning"
    # Nested keys as tuples for use in helpers
    TARGET_PCA_DF = ("target", "pca_df")
    TARGET_CONTRIB_DF = ("target", "contrib_df")
