# Copyright (C) 2024 David West Brown

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import altair as alt
import numpy as np
import pandas as pd
import polars as pl
import scipy
from sklearn import decomposition

def subset_pl(tok_pl, select_ids: list):
	token_subset = (
		tok_pl
		.with_columns(
			pl.col("doc_id").str.split_exact("_", 0)
			.struct.rename_fields(["cat_id"])
			.alias("id")
		)
		.unnest("id")
		.filter(pl.col("cat_id").is_in(select_ids))
		.drop("cat_id")
		)
	return(token_subset)

def frequency_tables_pl(tok_pl):
	
	def summarize_counts(df):
			df = (
				df
				.pivot(index="Token", on="doc_id", values="len", aggregate_function="sum")
				.with_columns(
					pl.all().exclude("Token").cast(pl.UInt32, strict=True)
				)
				# calculate range
				.with_columns(
					pl.sum_horizontal(pl.selectors.numeric().is_not_null())
					.alias("Range")
				)
				.with_columns(
					pl.selectors.numeric().fill_null(strategy="zero")
				)
				# normalize over total documents in corpus
				.with_columns(
						pl.col("Range").truediv(pl.sum_horizontal(pl.selectors.numeric().exclude("Range").is_not_null())).mul(100)
				)
				# calculate absolute frequency
				.with_columns(
					pl.sum_horizontal(pl.selectors.numeric().exclude("Range"))
					.alias("AF")
				)
				.sort("AF", descending = True)
				.select(["Token", "AF", "Range"])
				# calculate relative frequency
				.with_columns(
					pl.col("AF").truediv(pl.sum("AF")).mul(1000000)
					.alias("RF")
				)
				# format data
				.unnest("Token")
				.select(["Token", "Tag", "AF", "RF", "Range"])
				)
			return(df)
	
	# format tokens and sum by doc_id
	df_pos = (
		tok_pl
		.group_by(["doc_id", "pos_id", "pos_tag"], maintain_order = True)
		.agg(
			pl.col("token").str.concat("")
		)
		.with_columns(
			pl.col("token").str.to_lowercase().str.strip_chars())
		.filter(
			pl.col("pos_tag") != "Y"
		)
		.rename({"pos_tag": "Tag"})
		.rename({"token": "Token"})
		.group_by(["doc_id", "Token", "Tag"]).len()
		.with_columns(
			pl.struct(["Token", "Tag"])
		)
		.select(pl.exclude("Tag"))
		)
	
	df_pos = summarize_counts(df_pos).sort(["AF", "Token"], descending=[True, False])
   
	df_ds = (
		tok_pl
		.group_by(["doc_id", "ds_id", "ds_tag"], maintain_order = True)
		.agg(
			pl.col("token").str.concat("")
		)
		.with_columns(
			pl.col("token").str.to_lowercase().str.strip_chars())
		.filter(
			~(pl.col("token").str.contains("^[[[:punct:]] ]+$") & pl.col("ds_tag").str.contains("Untagged"))
		)
		.rename({"ds_tag": "Tag"})
		.rename({"token": "Token"})
		.group_by(["doc_id", "Token", "Tag"]).len()
		.with_columns(
			pl.struct(["Token", "Tag"])
		)
		.select(pl.exclude("Tag"))
		)
	
	df_ds = summarize_counts(df_ds).sort(["AF", "Token"], descending=[True, False])

	return(df_pos, df_ds)
    
def tag_tables_pl(tok_pl):
	
	def summarize_counts(df):
			df = (
				df
				.pivot(index="Tag", on="doc_id", values="len", aggregate_function="sum")
				.with_columns(
					pl.all().exclude("Tag").cast(pl.UInt32, strict=True)
				)
				# calculate range
				.with_columns(
					Range=pl.sum_horizontal(pl.selectors.numeric().is_not_null())
				)
 				.with_columns(
					pl.selectors.numeric().fill_null(strategy="zero")
				)
				# normalize over total documents in corpus
				.with_columns(
						pl.col("Range").truediv(pl.sum_horizontal(pl.selectors.numeric().exclude("Range").is_not_null())).mul(100)
				)
				# calculate absolute frequency
				.with_columns(
					pl.sum_horizontal(pl.selectors.numeric().exclude("Range"))
					.alias("AF")
				)
				.sort("AF", descending = True)
				.select(["Tag", "AF", "Range"])
				# calculate relative frequency
				.with_columns(
					pl.col("AF").truediv(pl.sum("AF")).mul(100)
					.alias("RF")
				)
			  .select(["Tag", "AF", "RF", "Range"])
				)
			return(df)
	
	# format tokens and sum by doc_id
	df_pos = (
		tok_pl
		.filter(pl.col("pos_tag") != "Y")
		.group_by(["doc_id", "pos_id", "pos_tag"], maintain_order = True)
		.first()
		.group_by(["doc_id", "pos_tag"]).len()
		.rename({"pos_tag": "Tag"})
		)
	
	df_pos = summarize_counts(df_pos).sort(["AF", "Tag"], descending=[True, False])
   
	df_ds = (
		tok_pl
		.filter(~(pl.col("token").str.contains("^[[[:punct:]] ]+$") & pl.col("ds_tag").str.contains("Untagged")))
		.group_by(["doc_id", "ds_id", "ds_tag"], maintain_order = True)
		.first()
		.group_by(["doc_id", "ds_tag"]).len()
		.rename({"ds_tag": "Tag"})
		)
	
	df_ds = summarize_counts(df_ds).sort(["AF", "Tag"], descending=[True, False])

	return(df_pos, df_ds)

def dtm_pl(tok_pl):
	
	df_pos = (
		tok_pl
		.filter(pl.col("pos_tag") != "Y")
		.group_by(["doc_id", "pos_id", "pos_tag"], maintain_order = True)
		.first()
		.group_by(["doc_id", "pos_tag"]).len()
		.rename({"pos_tag": "tag"})
		.with_columns(pl.col("len").sum().over('tag').alias('total'))
		.sort(["total", "doc_id"], descending=[True, False])
		.pivot(index="doc_id", on="tag", values="len", aggregate_function="sum")
		.fill_null(strategy="zero")
		)

	df_ds = (
		tok_pl
		.filter(~(pl.col("token").str.contains("^[[[:punct:]] ]+$") & pl.col("ds_tag").str.contains("Untagged")))
		.group_by(["doc_id", "ds_id", "ds_tag"], maintain_order = True)
		.first()
		.group_by(["doc_id", "ds_tag"]).len()
		.rename({"ds_tag": "tag"})
		.with_columns(pl.col("len").sum().over('tag').alias('total'))
		.sort(["total", "doc_id"], descending=[True, False])
		.pivot(index="doc_id", on="tag", values="len", aggregate_function="sum")
		.fill_null(strategy="zero")
	)

	return(df_pos, df_ds)

def collocations_pl(tok_pl, node_word, preceding=4, following=4, statistic='pmi', count_by='pos', node_tag=None):

	if count_by == 'pos':
		grouping_tag = "pos_tag"
		grouping_id = "pos_id"
		expr_filter = pl.col("pos_tag") != "Y"
	else:
		grouping_tag = "ds_tag"
		grouping_id = "ds_id"
		expr_filter = ~(pl.col("token").str.contains("^[[[:punct:]] ]+$") & pl.col("ds_tag").str.contains("Untagged"))

	if node_tag is None:
		expr = pl.col("token") == node_word.lower()
	else:
		expr = (pl.col("token") == node_word.lower()) & (pl.col(grouping_tag).str.starts_with(node_tag))

	look_around_token = [
		pl.col("token").shift(-i).alias(f"tok_lag_{i}") for i in range(-preceding, following + 1)
	]
	look_around_tag = [
		pl.col(grouping_tag).shift(-i).alias(f"tag_lag_{i}") for i in range(-preceding, following + 1)
	]

	total_df = (
		tok_pl
		.group_by(["doc_id", grouping_id, grouping_tag], maintain_order = True)
		.agg(
			pl.col("token").str.concat("")
			)
		.with_columns(
			pl.col("token").str.to_lowercase().str.strip_chars())
		.filter(expr_filter)
		.group_by(["token", grouping_tag]).len(name="Freq_Total")
		.rename({"token": "Token", grouping_tag: "Tag"})
	)

	token_total = sum(total_df.get_column("Freq_Total"))
	
	if node_tag is None:
		node_freq = total_df.filter(pl.col("Token") == node_word).get_column("Freq_Total").sum()
	else:
		node_freq = total_df.filter((pl.col("Token") == node_word.lower()) & (pl.col("Tag").str.starts_with(node_tag))).get_column("Freq_Total").sum()
			
	if node_freq == 0:
		coll_df = pl.DataFrame(schema=[("Token", pl.String), ("Tag", pl.String), ("Freq Span", pl.UInt32), ("Freq Total", pl.UInt32), ("MI", pl.Float64)])
		return(coll_df)

	if statistic=='pmi':
		mi_funct = pl.col("Freq_Span").truediv(token_total).log(base=2).sub(pl.col("Freq_Total").truediv(token_total).mul(node_freq).truediv(token_total).log(base=2))
	if statistic=='npmi':
		mi_funct = pl.col("Freq_Span").truediv(token_total).log(base=2).sub(pl.col("Freq_Total").truediv(token_total).mul(node_freq).truediv(token_total).log(base=2)).truediv(pl.col("Freq_Span").truediv(token_total).log(base=2).neg())
	if statistic=='pmi2':
		mi_funct = pl.col("Freq_Span").truediv(token_total).log(base=2).sub(pl.col("Freq_Total").truediv(token_total).mul(node_freq).truediv(token_total).log(base=2)).sub(pl.col("Freq_Span").truediv(token_total).log(base=2).mul(-1))
	if statistic=='pmi3':
		mi_funct = pl.col("Freq_Span").truediv(token_total).log(base=2).sub(pl.col("Freq_Total").truediv(token_total).mul(node_freq).truediv(token_total).log(base=2)).sub(pl.col("Freq_Span").truediv(token_total).log(base=2).mul(-2))

	coll_df = (
		tok_pl
		.group_by(["doc_id", grouping_id, grouping_tag], maintain_order = True)
		.agg(
			pl.col("token").str.concat("")
			)
		.with_columns(
			pl.col("token").str.to_lowercase().str.strip_chars())
		.filter(
			pl.col('token').str.contains("[a-z]")
			)
		.with_columns(
			look_around_token + look_around_tag
			)
		.filter(expr)
		#.drop(["tok_lag_0", "tag_lag_0"])
		.group_by("doc_id")
		.agg(
			pl.concat_list([f"tok_lag_{i}" for i in range(-preceding, following + 1)]).alias("span_tok"),
			pl.concat_list([f"tag_lag_{i}" for i in range(-preceding, following + 1)]).alias("span_tag")
			)
		.explode(["span_tok", "span_tag"])
		.with_columns(
				pre_node_tok=pl.col("span_tok").list.head(preceding),
				pre_node_tag=pl.col("span_tag").list.head(preceding)
			)
		.with_columns(
				post_node_tok=pl.col("span_tok").list.tail(following),
				post_node_tag=pl.col("span_tag").list.tail(following)
			)
		.drop(["span_tok", "span_tag"])
		.with_columns(
			Token=pl.col("pre_node_tok").list.concat("post_node_tok"),
			Tag=pl.col("pre_node_tag").list.concat("post_node_tag")
			)
		.select(["Token", "Tag"])
		.explode(["Token", "Tag"])
		.group_by(["Token", "Tag"]).len(name="Freq_Span")
		.sort("Freq_Span")
		.join(total_df, on=["Token", "Tag"])
		.with_columns(
			MI=mi_funct
			)
		.rename({"Freq_Span": "Freq Span", "Freq_Total": "Freq Total"})
		.sort("MI", "Token", descending=[True, False])
	)
	
	return(coll_df)

def keyness_pl(target_pl, reference_pl, correct=False, tags_only=False, threshold=.01):

	total_target = target_pl.get_column("AF").sum()
	total_reference = reference_pl.get_column("AF").sum()
	total_tokens = total_target + total_reference

	if correct == False:
		correction_tar = pl.col("AF")
		correction_ref = pl.col("AF_Ref")
	if correct == True:
		correction_tar = pl.col("AF").sub(.5 * pl.col("AF").sub((pl.col("AF").add(pl.col("AF_Ref")).mul(total_target / total_tokens))).abs().truediv(pl.col("AF").sub((pl.col("AF").add(pl.col("AF_Ref")).mul(total_target / total_tokens)))))
		correction_ref = pl.col("AF_Ref").add(.5 * pl.col("AF").sub((pl.col("AF").add(pl.col("AF_Ref")).mul(total_target / total_tokens))).abs().truediv(pl.col("AF").sub((pl.col("AF").add(pl.col("AF_Ref")).mul(total_target / total_tokens)))))
	
	if tags_only == False:
		kw_df = target_pl.join(reference_pl, on=["Token", "Tag"], how="full", coalesce=True, suffix="_Ref").fill_null(strategy="zero")
	if tags_only == True:
		kw_df = target_pl.join(reference_pl, on="Tag", how="full", coalesce=True, suffix="_Ref").fill_null(strategy="zero")

	kw_df = (
		kw_df
		.with_columns(
			pl.when(pl.col("AF").sub((pl.col("AF").add(pl.col("AF_Ref")).mul(total_target / total_tokens))).abs() > .25)
			.then(correction_tar)
			.otherwise(pl.col("AF"))
			.alias("AF_Yates")
			)
		.with_columns(
			pl.when(pl.col("AF").sub((pl.col("AF").add(pl.col("AF_Ref")).mul(total_target / total_tokens))).abs() > .25)
			.then(correction_ref)
			.otherwise(pl.col("AF_Ref"))
			.alias("AF_Ref_Yates")
			)
		.with_columns(
			pl.when(pl.col("AF_Yates") > 0)
			.then(
				pl.col("AF_Yates").mul(pl.col("AF_Yates").truediv(pl.col("AF_Yates").add(pl.col("AF_Ref")).mul(total_target / total_tokens)).log())
			)
			.otherwise(0)
			.alias("L1")
			)
		.with_columns(
			pl.when(pl.col("AF_Ref_Yates") > 0)
			.then(
				pl.col("AF_Ref_Yates").mul(pl.col("AF_Ref_Yates").truediv(pl.col("AF_Yates").add(pl.col("AF_Ref_Yates")).mul(total_reference / total_tokens)).log())
			)
			.otherwise(0)
			.alias("L2")
			)
		.with_columns(
			pl.when(pl.col("RF") > pl.col("RF_Ref"))
			.then(
				pl.col("L1").add(pl.col("L2")).mul(2).abs()
			)
			.otherwise(
				pl.col("L1").add(pl.col("L2")).mul(2).abs().neg()
			)
			.alias("LL")
		)
		.with_columns(
			pl.when(pl.col("AF_Ref") == 0)
			.then(
				pl.col("AF").truediv(total_target).truediv(.5 / total_reference).log(base=2)
			)
			.when(pl.col("AF") == 0)
			.then(
				pl.col("AF_Ref").truediv(total_reference).truediv(.5 / total_target).log(base=2).neg()
			)
			.otherwise(
				pl.col("AF").truediv(total_target).truediv(pl.col("AF_Ref").truediv(total_reference)).log(base=2)
			)
			.alias("LR")
		)
		.with_columns(
			pl.col("LL").abs().map_elements(lambda x: scipy.stats.distributions.chi2.sf(x, 1), return_dtype=pl.Float64)
			.alias("PV")
		)
		.sort("LL", descending=True)
		.filter(pl.col("PV") < threshold)
	)
	if tags_only == False:
		return(kw_df.select(["Token", "Tag", "LL", "LR", "PV", "RF", "RF_Ref", "AF", "AF_Ref", "Range", "Range_Ref"]))
		
	if tags_only == True:
		return(kw_df.select(["Tag", "LL", "LR", "PV", "RF", "RF_Ref", "AF", "AF_Ref", "Range", "Range_Ref"]))
	

def ngrams_by_token_pl(tok_pl, node_word: str, node_position, span, search_type, count_by='pos'):
	
	if count_by == 'pos':
		grouping_tag = "pos_tag"
		grouping_id = "pos_id"
		expr_filter = pl.col("pos_tag") != "Y"
	else:
		grouping_tag = "ds_tag"
		grouping_id = "ds_id"
		expr_filter = ~(pl.col("token").str.contains("^[[[:punct:]] ]+$") & pl.col("ds_tag").str.contains("Untagged"))
	
	if search_type == "fixed":
		expr = pl.col("token") == node_word.lower()
	elif search_type == "starts_with":
		expr = pl.col("token").str.starts_with(node_word.lower())
	elif search_type == "ends_with":
		expr = pl.col("token").str.ends_with(node_word.lower())
	elif search_type == "contains":
		expr = pl.col("token").str.contains(node_word.lower())
	
	preceding = node_position - 1
	following = span - node_position
	
	look_around_token = [
		pl.col("token").shift(-i).alias(f"tok_lag_{i}") for i in range(-preceding, following + 1)
	]
	look_around_tag = [
		pl.col(grouping_tag).shift(-i).alias(f"tag_lag_{i}") for i in range(-preceding, following + 1)
	]

	rename_tokens = [
		 pl.col('ngram').struct.rename_fields([f'Token_{i + 1}' for i in range(span)])
	]
	rename_tags = [
		 pl.col('tags').struct.rename_fields([f'Tag_{i + 1}' for i in range(span)])
	]
	
	ngram_df = (
		tok_pl
		.group_by(["doc_id", grouping_id, grouping_tag], maintain_order = True)
		.agg(
			pl.col("token").str.concat("")
			)
		.filter(expr_filter)
		.with_columns(pl.col("token").len().alias("total"))
		.with_columns(
			pl.col("token").str.to_lowercase().str.strip_chars())
		.with_columns(
			look_around_token + look_around_tag
			)
		.filter(expr)
		.group_by("doc_id", "total")
		.agg(
			pl.concat_list([f"tok_lag_{i}" for i in range(-preceding, following + 1)]).alias("ngram"),
			pl.concat_list([f"tag_lag_{i}" for i in range(-preceding, following + 1)]).alias("tags")
			)
		.explode(["ngram", "tags"])
	)

	if ngram_df.height == 0:
		return(ngram_df)
	
	else:
		ngram_df = (
		ngram_df
		.with_columns(
			pl.col(["ngram", "tags"]).list.to_struct()
			)
		.group_by(["doc_id", "total", "ngram", "tags"]).len().sort("len", descending=True)
		.with_columns(
			pl.struct(["ngram", "tags"])
			)
		.select(pl.exclude("tags"))
		.pivot(index=["ngram", "total"], on="doc_id", values="len", aggregate_function="sum")
		.with_columns(
			pl.all().exclude("ngram").cast(pl.UInt32, strict=True)
			)
		# calculate range
		.with_columns(
			pl.sum_horizontal(pl.selectors.numeric().exclude("total").is_not_null())
			.alias("Range")
			)
		.with_columns(
			pl.selectors.numeric().fill_null(strategy="zero")
			)
		# normalize over total documents in corpus
		.with_columns(
			pl.col("Range").truediv(pl.sum_horizontal(pl.selectors.numeric().exclude(["Range", "total"]).is_not_null())).mul(100)
			)
		# calculate absolute frequency
		.with_columns(
			pl.sum_horizontal(pl.selectors.numeric().exclude(["Range", "total"]))
			.alias("AF")
			)
		.sort("AF", descending = True)
		# calculate relative frequency
		.with_columns(
			pl.col("AF").truediv(pl.col("total")).mul(1000000)
			.alias("RF")
			)
		.select(["ngram", "AF", "RF", "Range"])
		.unnest("ngram")
		.with_columns(
			rename_tokens + rename_tags
			)
		.unnest(["ngram", "tags"])
		)
		return ngram_df

def ngrams_by_tag_pl(tok_pl, tag: str, node_position, span, count_by='pos'):
			
	if count_by == 'pos':
		grouping_tag = "pos_tag"
		grouping_id = "pos_id"
		expr = pl.col("pos_tag") == tag
		expr_filter = pl.col("pos_tag") != "Y"
	else:
		grouping_tag = "ds_tag"
		grouping_id = "ds_id"
		expr = pl.col("ds_tag") == tag
		expr_filter = ~(pl.col("token").str.contains("^[[[:punct:]] ]+$") & pl.col("ds_tag").str.contains("Untagged"))
	
	preceding = node_position - 1
	following = span - node_position
	
	look_around_token = [
		pl.col("token").shift(-i).alias(f"tok_lag_{i}") for i in range(-preceding, following + 1)
	]
	look_around_tag = [
		pl.col(grouping_tag).shift(-i).alias(f"tag_lag_{i}") for i in range(-preceding, following + 1)
	]

	rename_tokens = [
		 pl.col('ngram').struct.rename_fields([f'Token_{i + 1}' for i in range(span)])
	]
	rename_tags = [
		 pl.col('tags').struct.rename_fields([f'Tag_{i + 1}' for i in range(span)])
	]
	
	ngram_df = (
		tok_pl
		.group_by(["doc_id", grouping_id, grouping_tag], maintain_order = True)
		.agg(
			pl.col("token").str.concat("")
			)
		.filter(expr_filter)
		.with_columns(pl.col("token").len().alias("total"))
		.with_columns(
			pl.col("token").str.to_lowercase().str.strip_chars())
		.with_columns(
			look_around_token + look_around_tag
			)
		.filter(expr)
		.group_by("doc_id", "total")
		.agg(
			pl.concat_list([f"tok_lag_{i}" for i in range(-preceding, following + 1)]).alias("ngram"),
			pl.concat_list([f"tag_lag_{i}" for i in range(-preceding, following + 1)]).alias("tags")
			)
		.explode(["ngram", "tags"])
	)

	if ngram_df.height == 0:
		return(ngram_df)
	
	else:
		ngram_df = (
		ngram_df
		.with_columns(
			pl.col(["ngram", "tags"]).list.to_struct()
			)
		.group_by(["doc_id", "total", "ngram", "tags"]).len().sort("len", descending=True)
		.with_columns(
			pl.struct(["ngram", "tags"])
			)
		.select(pl.exclude("tags"))
		.pivot(index=["ngram", "total"], on="doc_id", values="len", aggregate_function="sum")
		.with_columns(
			pl.all().exclude("ngram").cast(pl.UInt32, strict=True)
			)
		# calculate range
		.with_columns(
			pl.sum_horizontal(pl.selectors.numeric().exclude("total").is_not_null())
			.alias("Range")
			)
		.with_columns(
			pl.selectors.numeric().fill_null(strategy="zero")
			)
		# normalize over total documents in corpus
		.with_columns(
			pl.col("Range").truediv(pl.sum_horizontal(pl.selectors.numeric().exclude(["Range", "total"]).is_not_null())).mul(100)
			)
		# calculate absolute frequency
		.with_columns(
			AF=pl.sum_horizontal(pl.selectors.numeric().exclude(["Range", "total"]))
			)
		.sort("AF", descending = True)
		# calculate relative frequency
		.with_columns(
			pl.col("AF").truediv(pl.col("total")).mul(1000000)
			.alias("RF")
			)
		.select(["ngram", "AF", "RF", "Range"])
		.unnest("ngram")
		.with_columns(
			rename_tokens + rename_tags
			)
		.unnest(["ngram", "tags"])
		)
		return ngram_df

def ngrams_pl(tok_pl, span, count_by='pos', min_frequency=10):
	
	if count_by == 'pos':
		grouping_tag = "pos_tag"
		grouping_id = "pos_id"
		expr_filter = pl.col("pos_tag") != "Y"
	else:
		grouping_tag = "ds_tag"
		grouping_id = "ds_id"
		expr_filter = ~(pl.col("token").str.contains("^[[[:punct:]] ]+$") & pl.col("ds_tag").str.contains("Untagged"))
		
	look_around_token = [
		pl.col("token").shift(-i).alias(f"tok_lag_{i}") for i in range(span)
    ]
	look_around_tag = [
		pl.col(grouping_tag).shift(-i).alias(f"tag_lag_{i}") for i in range(span)
    ]

	rename_tokens = [
		 pl.col('ngram').struct.rename_fields([f'Token_{i + 1}' for i in range(span)])
	]
	rename_tags = [
		 pl.col('tags').struct.rename_fields([f'Tag_{i + 1}' for i in range(span)])
	]
	
	ngram_df = (
		tok_pl
		.group_by(["doc_id", grouping_id, grouping_tag], maintain_order = True)
		.agg(
			pl.col("token").str.concat("")
			)
		.filter(expr_filter)
		.with_columns(pl.col("token").len().alias("total"))
		.with_columns(
			pl.col("token").str.to_lowercase().str.strip_chars())
		.with_columns(
			look_around_token + look_around_tag
			)
		.group_by("doc_id", "total")
		.agg(
			pl.concat_list([f"tok_lag_{i}" for i in range(span)]).alias("ngram"),
			pl.concat_list([f"tag_lag_{i}" for i in range(span)]).alias("tags")
			)
		.explode(["ngram", "tags"])
		.with_columns(
			pl.col(["ngram", "tags"]).list.to_struct()
			)
		.group_by(["doc_id", "total", "ngram", "tags"]).len().sort("len", descending=True)
		.with_columns(
			pl.struct(["ngram", "tags"])
			)
		.select(pl.exclude("tags"))
		.pivot(index=["ngram", "total"], on="doc_id", values="len", aggregate_function="sum")
		.with_columns(
			pl.all().exclude("ngram").cast(pl.UInt32, strict=True)
			)
		# calculate range
		.with_columns(
			pl.sum_horizontal(pl.selectors.numeric().exclude("total").is_not_null())
			.alias("Range")
			)
		.with_columns(
			pl.selectors.numeric().fill_null(strategy="zero")
			)
		# normalize over total documents in corpus
		.with_columns(
				pl.col("Range").truediv(pl.sum_horizontal(pl.selectors.numeric().exclude(["Range", "total"]).is_not_null())).mul(100)
			)
		# calculate absolute frequency
		.with_columns(
			pl.sum_horizontal(pl.selectors.numeric().exclude(["Range", "total"]))
			.alias("AF")
			)
		.sort("AF", descending = True)
        # calculate relative frequency
		.with_columns(
			pl.col("AF").truediv(pl.col("total")).mul(1000000)
			.alias("RF")
			)
		.select(["ngram", "AF", "RF", "Range"])
		.unnest("ngram")
		.with_columns(
			rename_tokens + rename_tags
			)
		.unnest(["ngram", "tags"])
		.sort(["AF", "Token_1", "Token_2"], descending=[True, False, False])
		.filter(
               pl.col('RF') >= min_frequency
			)
		)
	
	return ngram_df

def kwic_pl(tok_pl, node_word: str, search_type="fixed", ignore_case=True):
	
	if search_type == "fixed" and ignore_case == True:
		expr = pl.col("token").str.to_lowercase().str.strip_chars() == node_word.lower()
	if search_type == "fixed" and ignore_case == False:
		expr = pl.col("token").str.strip_chars() == node_word
	elif search_type == "starts_with" and ignore_case == True:
		expr = pl.col("token").str.to_lowercase().str.strip_chars().str.starts_with(node_word.lower())
	elif search_type == "starts_with" and ignore_case == False:
		expr = pl.col("token").str.strip_chars().str.starts_with(node_word)
	elif search_type == "ends_with" and ignore_case == True:
		expr = pl.col("token").str.to_lowercase().str.strip_chars().str.ends_with(node_word.lower())
	elif search_type == "ends_with" and ignore_case == False:
		expr = pl.col("token").str.ends_with(node_word)
	elif search_type == "contains" and ignore_case == True:
		expr = pl.col("token").str.to_lowercase().str.strip_chars().str.contains(node_word.lower())
	elif search_type == "contains" and ignore_case == False:
		expr = pl.col("token").str.strip_chars().str.contains(node_word)

	
	preceding = 7
	following = 7
	
	look_around_token = [
		pl.col("token").shift(-i).alias(f"tok_lag_{i}") for i in range(-preceding, following + 1)
    ]
	
	kwic_df = (
		tok_pl
		.group_by(["doc_id", "pos_id"], maintain_order = True)
		.agg(
			pl.col("token").str.concat("")
			)
		.with_columns(
			look_around_token
			)
		.filter(expr)
		.group_by("doc_id")
		.agg(
			pl.concat_list([f"tok_lag_{i}" for i in range(-preceding, following + 1)]).alias("node")
			)
		.explode("node")
		.with_columns(
			pre_node=pl.col("node").list.head(7)
		)
		.with_columns(
			post_node=pl.col("node").list.tail(7)
		)
		.with_columns(
			pl.col("node").list.get(7)
		)
		.with_columns(
			pl.col("pre_node").list.join("")
		)
		.with_columns(
			pl.col("post_node").list.join("")
		)
		.select(["doc_id", "pre_node", "node", "post_node"])
		.sort("doc_id")
        .rename({"doc_id": "Doc ID", "pre_node": "Pre-Node", "node": "Node", "post_node": "Post-Node"})
	)
	
	return kwic_df

def split_corpus(tok, tar_list, ref_list):
	tar_docs = {key: value for key, value in tok.items() if key.startswith(tuple(tar_list))}
	ref_docs = {key: value for key, value in tok.items() if key.startswith(tuple(ref_list))}
	tar_ndocs = len(tar_docs)
	ref_ndocs = len(ref_docs)
	#get target counts
	tar_tok = list(tar_docs.values())
	tar_tags = []
	for i in range(0,len(tar_tok)):
		tags = [x[1] for x in tar_tok[i]]
		tar_tags.append(tags)
	tar_tags = [x for xs in tar_tags for x in xs]
	tar_tokens = len(tar_tags)
	tar_words = len([x for x in tar_tags if not x.startswith('Y')])
	
	#get reference counts
	ref_tok = list(ref_docs.values())
	ref_tags = []
	for i in range(0,len(ref_tok)):
		tags = [x[1] for x in ref_tok[i]]
		ref_tags.append(tags)
	ref_tags = [x for xs in ref_tags for x in xs]
	ref_tokens = len(ref_tags)
	ref_words = len([x for x in ref_tags if not x.startswith('Y')])
	return tar_docs, ref_docs, tar_words, ref_words, tar_tokens, ref_tokens, tar_ndocs, ref_ndocs

def dtm_weight_pl(dtm_pl, scheme="prop"):
	if scheme == "prop":
		expr = pl.selectors.numeric().truediv(pl.sum_horizontal(pl.selectors.numeric()))
	if scheme == "scale":
		expr = pl.selectors.numeric().sub(pl.selectors.numeric().mean()).truediv(pl.selectors.numeric().std())
	if scheme == "tfidf":
		expr = pl.selectors.numeric().mul(pl.col("doc_id").len().truediv(pl.selectors.numeric().gt(0).sum()).log10())

	weighted_df = (
		dtm_pl
		.with_columns(
			expr
		)
	)
	return(weighted_df)

def dtm_simplify_pl(dtm_pl):
 simple_df = (
	dtm_pl
	.unpivot(pl.selectors.numeric(), index="doc_id")
	.with_columns(
		pl.col("variable")
		.str.replace('^NN\S*$', '#NounCommon')
		.str.replace('^VV\S*$', '#VerbLex')
		.str.replace('^J\S*$', '#Adjective')
		.str.replace('^R\S*$', '#Adverb')
		.str.replace('^P\S*$', '#Pronoun')
		.str.replace('^I\S*$', '#Preposition')
		.str.replace('^C\S*$', '#Conjunction')
		.str.replace('^N\S*$', '#NounOther')
		.str.replace('^VB\S*$', '#VerbBe')
		.str.replace('^V\S*$', '#VerbOther')
	)
	.with_columns(
		pl.when(pl.col("variable").str.starts_with("#"))
		.then(pl.col("variable"))
		.otherwise(pl.col("variable").str.replace('^\S+$', '#Other'))
		)
	.with_columns(
		pl.col("variable").str.replace("#", "")
	)
	.group_by(["doc_id", "variable"], maintain_order = True).sum()
	.pivot(index="doc_id", on="variable", values="value")
	)
	
 return(simple_df)

def freq_simplify_pl(df_pl):
 simple_df = (
	df_pl
	.with_columns(
		pl.selectors.starts_with("Tag")
		.str.replace('^NN\S*$', '#NounCommon')
		.str.replace('^VV\S*$', '#VerbLex')
		.str.replace('^J\S*$', '#Adjective')
		.str.replace('^R\S*$', '#Adverb')
		.str.replace('^P\S*$', '#Pronoun')
		.str.replace('^I\S*$', '#Preposition')
		.str.replace('^C\S*$', '#Conjunction')
		.str.replace('^N\S*$', '#NounOther')
		.str.replace('^VB\S*$', '#VerbBe')
		.str.replace('^V\S*$', '#VerbOther')
	)
	.with_columns(
		pl.when(pl.selectors.starts_with("Tag").str.starts_with("#"))
		.then(pl.selectors.starts_with("Tag"))
		.otherwise(pl.selectors.starts_with("Tag").str.replace('^\S+$', '#Other'))
		)
	.with_columns(
		pl.selectors.starts_with("Tag").str.replace("#", "")
	)
	)
	
 return(simple_df)


def tags_simplify_pl(dtm_pl):
	simple_df = (
		dtm_pl
		.transpose(include_header=True, header_name="Tag", column_names="doc_id")
		.with_columns(
				pl.sum_horizontal(pl.selectors.numeric() > 0)
				.alias("Range")
		)
	.with_columns(
		pl.col("Range").truediv(pl.sum_horizontal(pl.selectors.numeric().exclude("Range").is_not_null())).mul(100)
		)
		# calculate absolute frequency
		.with_columns(
			pl.sum_horizontal(pl.selectors.numeric().exclude("Range"))
			.alias("AF")
		)
		.sort("AF", descending = True)
		.select(["Tag", "AF", "Range"])
		# calculate relative frequency
		.with_columns(
			pl.col("AF").truediv(pl.sum("AF")).mul(100)
			.alias("RF")
		)
		.select(["Tag", "AF", "RF", "Range"])
		)
	return(simple_df)


def pca_contributions(dtm, doccats):
	df = dtm.set_index('doc_id')
	n = min(len(df.index), len(df.columns))
	pca = decomposition.PCA(n_components=n)
	pca_result = pca.fit_transform(df.values)
	pca_df = pd.DataFrame(pca_result)
	pca_df.columns = ['PC' + str(col + 1) for col in pca_df.columns]
						
	sdev = pca_df.std(ddof=0)
	contrib = []
	for i in range(0, len(sdev)):
		coord = pca.components_[i] * sdev.iloc[i]
		polarity = np.divide(coord, abs(coord))
		coord = np.square(coord)
		coord = np.divide(coord, sum(coord))*100
		coord = np.multiply(coord, polarity)				
		contrib.append(coord)
	contrib_df =  pd.DataFrame(contrib).transpose()
	contrib_df.columns = ['PC' + str(col + 1) for col in contrib_df.columns]
	contrib_df['Tag'] = df.columns
	if len(doccats) > 0:
		pca_df['Group'] = doccats
	pca_df['doc_id'] = list(df.index)		
	ve = np.array(pca.explained_variance_ratio_).tolist()
	return pca_df, contrib_df, ve

def update_pca_plot(coord_data, contrib_data, variance, pca_idx):
	pca_x = coord_data.columns[pca_idx - 1]
	pca_y = coord_data.columns[pca_idx]
	
	mean_x = contrib_data[pca_x].abs().mean()
	mean_y = contrib_data[pca_y].abs().mean()
	contrib_x = contrib_data[contrib_data[pca_x].abs() > mean_x]
	contrib_x[pca_x] = contrib_x[pca_x]
	contrib_x.sort_values(by=pca_x, ascending=False, inplace=True)
	contrib_x_values = contrib_x.loc[:,pca_x].tolist()
	contrib_x_values = ['%.2f' % x for x in contrib_x_values]
	contrib_x_values = [x + "%" for x in contrib_x_values]
	contrib_x_tags = contrib_x.loc[:,"Tag"].tolist()
	contrib_x = list(zip(contrib_x_tags, contrib_x_values))
	contrib_x = list(map(', '.join, contrib_x))
	contrib_x = '; '.join(contrib_x)
	
	contrib_y = contrib_data[contrib_data[pca_y].abs() > mean_y]
	contrib_y[pca_y] = contrib_y[pca_y]
	contrib_y.sort_values(by=pca_y, ascending=False, inplace=True)
	contrib_y_values = contrib_y.loc[:,pca_y].tolist()
	contrib_y_values = ['%.2f' % y for y in contrib_y_values]
	contrib_y_values = [y + "%" for y in contrib_y_values]
	contrib_y_tags = contrib_y.loc[:,"Tag"].tolist()
	contrib_y = list(zip(contrib_y_tags, contrib_y_values))
	contrib_y = list(map(', '.join, contrib_y))
	contrib_y = '; '.join(contrib_y)
	
	contrib_1 = contrib_data[contrib_data[pca_x].abs() > 0]
	contrib_1[pca_x] = contrib_1[pca_x].div(100)
	contrib_1.sort_values(by=pca_x, ascending=True, inplace=True)
	
	contrib_2 = contrib_data[contrib_data[pca_y].abs() > 0]
	contrib_2[pca_y] = contrib_2[pca_y].div(100)	
	contrib_2.sort_values(by=pca_y, ascending=True, inplace=True)
	
	ve_1 = "{:.2%}".format(variance[pca_idx - 1])
	ve_2 = "{:.2%}".format(variance[pca_idx])

	cp_1 = alt.Chart(contrib_1, height={"step": 24}).mark_bar(size=12).encode(
				x=alt.X(pca_x, axis=alt.Axis(format='%')), 
				y=alt.Y('Tag', sort='-x', title=None, axis=alt.Axis(labelLimit=200)),
	 			tooltip=[
				alt.Tooltip('Tag'),
     	  		 alt.Tooltip(pca_x, title="Cont.", format='.2%')
    			])
	
	cp_2 = alt.Chart(contrib_2, height={"step": 24}).mark_bar(size=12).encode(
				x=alt.X(pca_y, 
				axis=alt.Axis(format='%')), 
				y=alt.Y('Tag', sort='-x', title=None, axis=alt.Axis(labelLimit=200)),
				tooltip=[
				alt.Tooltip('Tag'),
				alt.Tooltip(pca_y, title="Cont.", format='.2%')
				])
		
	return(cp_1, cp_2, pca_x, pca_y, contrib_x, contrib_y, ve_1, ve_2)
	
def correlation(df, x, y):
	cc = scipy.stats.stats.pearsonr(df[x], df[y])
	cc_r = round(cc.statistic, 3)
	cc_p = round(cc.pvalue, 5)
	cc_df = len(df.index) - 2
	return cc_df, cc_r, cc_p

def boxplots_pl(dtm_pl, box_vals, grp_a = None, grp_b = None):

	df_plot = (
		dtm_pl
		.unpivot(pl.selectors.numeric(), index="doc_id", variable_name="Tag", value_name="RF")
		.with_columns(pl.col("RF").mul(100))
		.filter(pl.col("Tag").is_in(box_vals))
		.with_columns(
			pl.col("doc_id").str.split_exact("_", 0)
			.struct.rename_fields(["cat_id"])
			.alias("id")
			)
		.unnest("id")
	)

	if grp_a is None and grp_b is None:	

		df_plot = (df_plot
			.drop("cat_id")
			.with_columns(pl.median("RF").over("Tag").alias("Median"))
			.sort(["Median", "Tag"], descending=[True, False])
			)
		return(df_plot)

	if grp_a is not None and grp_b is not None:
		grp_a_str = ", ".join(str(x) for x in grp_a)
		grp_b_str = ", ".join(str(x) for x in grp_b)

		df_plot = (df_plot
			.with_columns(
				pl.when(pl.col("cat_id").is_in(grp_a))
				.then(pl.lit(grp_a_str))
				.when(pl.col("cat_id").is_in(grp_b))
				.then(pl.lit(grp_b_str))
				.otherwise(pl.lit("Other"))
				.alias("Group")
			)
			.drop("cat_id")
			.filter(pl.col("Group") != "Other")
			.with_columns(pl.median("RF").over("Group", "Tag").alias("Median"))
			.sort(["Median", "Tag"], descending=[True, False])
			)
		return(df_plot)
	
def scatterplots_pl(dtm_pl, axis_vals: list):

	df_plot = (
		dtm_pl
		.unpivot(pl.selectors.numeric(), index="doc_id", variable_name="Tag", value_name="AF")
		.with_columns(
			pl.when(pl.col("Tag").is_in(axis_vals))
			.then(pl.col("Tag"))
			.otherwise(pl.lit("Other"))
			.alias("Tag_Sort")
		)
		.with_columns(
			pl.col("doc_id").str.split_exact("_", 0)
			.struct.rename_fields(["cat_id"])
			.alias("id")
			)
		.unnest("id")
		.drop("cat_id", "Tag")
		.group_by(["doc_id", "Group", "Tag_Sort"]).sum()
		.with_columns(pl.col("AF").truediv(pl.sum("AF").over("doc_id")).mul(100).alias("RF"))
		.filter(pl.col("Tag_Sort") != "Other")
		.rename({"Tag_Sort": "Tag"})
		.sort("doc_id", "Tag")
		.pivot("Tag", index=["doc_id", "Group"], values="RF")
	)

	return(df_plot)


def html_build_pl(tok_pl, doc_key):
	html_pos = (
		tok_pl
		.filter(pl.col("doc_id") == doc_key)
		.group_by(["pos_id", "pos_tag"], maintain_order = True)
		.agg(pl.col("token").str.concat(""))
		.with_columns(pl.col("token").str.extract("(\s)$")
					.alias("ws"))
		.with_columns(pl.col("token").str.strip_chars())
		.with_columns(pl.col("token").str.len_chars()
						.alias("tag_end"))
		.with_columns(pl.col("tag_end").shift(1, fill_value=0)
					.alias("tag_start"))
		.with_columns(pl.col("tag_end").cum_sum())
		.with_columns(pl.col("tag_start").cum_sum())
		.with_columns(pl.col("ws").fill_null(""))
		.with_columns(
			pl.when(pl.col("pos_tag") != "Y")
			.then(pl.concat_str(pl.col("token"), pl.lit("</span>")))
			.otherwise(pl.col("token"))
			.alias("token_html"))
		.with_columns(
			pl.when(pl.col("pos_tag") != "Y")
			.then(pl.concat_str(pl.lit('<span class="'), pl.col("pos_tag"), pl.lit('">')))
			.otherwise(pl.lit(""))
			.alias("tag_html"))
		.with_columns(pl.concat_str(pl.col("tag_html"), pl.col("token_html"), pl.col("ws")).alias("Text"))
		.with_columns(pl.lit(doc_key).alias("doc_id"))
		.rename({"pos_tag": "Tag"})
		.select("doc_id", "token", "Tag", "tag_start", "tag_end", "Text")
	)

	html_simple = (
		tok_pl
		.filter(pl.col("doc_id") == doc_key)
		.group_by(["pos_id", "pos_tag"], maintain_order = True)
		.agg(pl.col("token").str.concat(""))
		.with_columns(pl.col("pos_tag")
		.str.replace('^NN\S*$', '#NounCommon')
		.str.replace('^VV\S*$', '#VerbLex')
		.str.replace('^J\S*$', '#Adjective')
		.str.replace('^R\S*$', '#Adverb')
		.str.replace('^P\S*$', '#Pronoun')
		.str.replace('^I\S*$', '#Preposition')
		.str.replace('^C\S*$', '#Conjunction')
		.str.replace('^N\S*$', '#NounOther')
		.str.replace('^VB\S*$', '#VerbBe')
		.str.replace('^V\S*$', '#VerbOther'))
		.with_columns(
			pl.when(pl.col("pos_tag").str.starts_with("#"))
			.then(pl.col("pos_tag"))
			.otherwise(pl.col("pos_tag").str.replace('^\S+$', '#Other')))
		.with_columns(pl.col("pos_tag").str.replace("#", ""))
		.with_columns(pl.col("token").str.extract("(\s)$")
					.alias("ws"))
		.with_columns(pl.col("token").str.strip_chars())
		.with_columns(pl.col("token").str.len_chars()
						.alias("tag_end"))
		.with_columns(pl.col("tag_end").shift(1, fill_value=0)
					.alias("tag_start"))
		.with_columns(pl.col("tag_end").cum_sum())
		.with_columns(pl.col("tag_start").cum_sum())
		.with_columns(pl.col("ws").fill_null(""))
		.with_columns(
			pl.when(pl.col("pos_tag") != "Other")
			.then(pl.concat_str(pl.col("token"), pl.lit("</span>")))
			.otherwise(pl.col("token"))
			.alias("token_html"))
		.with_columns(
			pl.when(pl.col("pos_tag") != "Other")
			.then(pl.concat_str(pl.lit('<span class="'), pl.col("pos_tag"), pl.lit('">')))
			.otherwise(pl.lit(""))
			.alias("tag_html"))
		.with_columns(pl.concat_str(pl.col("tag_html"), pl.col("token_html"), pl.col("ws")).alias("Text"))
		.with_columns(pl.lit(doc_key).alias("doc_id"))
		.rename({"pos_tag": "Tag"})
		.select("doc_id", "token", "Tag", "tag_start", "tag_end", "Text")
	)

	html_ds = (
		tok_pl
		.filter(pl.col("doc_id") == doc_key)
		.group_by(["ds_id", "ds_tag"], maintain_order = True)
		.agg(pl.col("token").str.concat(""))
		.with_columns(pl.col("token").str.extract("(\s)$")
					.alias("ws"))
		.with_columns(pl.col("token").str.strip_chars())
		.with_columns(pl.col("token").str.len_chars()
						.alias("tag_end"))
		.with_columns(pl.col("tag_end").shift(1, fill_value=0)
					.alias("tag_start"))
		.with_columns(pl.col("tag_end").cum_sum())
		.with_columns(pl.col("tag_start").cum_sum())
		.with_columns(pl.col("ws").fill_null(""))
		.with_columns(
			pl.when(pl.col("ds_tag") != "Untagged")
			.then(pl.concat_str(pl.col("token"), pl.lit("</span>")))
			.otherwise(pl.col("token"))
			.alias("token_html"))
		.with_columns(
			pl.when(pl.col("ds_tag") != "Untagged")
			.then(pl.concat_str(pl.lit('<span class="'), pl.col("ds_tag"), pl.lit('">')))
			.otherwise(pl.lit(""))
			.alias("tag_html"))
		.with_columns(pl.concat_str(pl.col("tag_html"), pl.col("token_html"), pl.col("ws")).alias("Text"))
		.with_columns(pl.lit(doc_key).alias("doc_id"))
		.rename({"ds_tag": "Tag"})
		.select("doc_id", "token", "Tag", "tag_start", "tag_end", "Text")
	)

	return(html_pos, html_simple, html_ds)

    