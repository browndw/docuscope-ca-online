# Copyright (C) 2023 David West Brown

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st

import categories as _categories

CATEGORY = _categories.HELP
TITLE = "Getting Started"
KEY_SORT = 12

def main():
	st.markdown("""
		Using most of the apps requires **only** that your upload and process a **target corpus**.
		Some, however, require additional data or will give you more options if you process additional data:
		
		* The **Compare Corpora** tool requires a **reference corpus**.
		* The **Compare Corpus Parts** tool requires processing **metadata** from the file names.
		* The **Advanced Plotting** tool will have more options if you've processed **metadata** from the file names.
		""")
		
	st.markdown("### :dart: Peparing a target corpus (Required)")
	
	st.markdown("""
		A corpus is simply an electronically searchable collection of texts.
		For this tool, a corpus must be made up of plain text or (**.txt**) files.
		If you are building your own corpus, it is strongly recommended that you use a plain
		text editor like **BBEdit** or **Sublime Text** and not something like Word. 
		""")
	st.markdown("##### Naming files (Optional)")

	st.markdown("""
	| :thumbsup: It is best practice not to include spaces in your file names. |
	|--------------------------------------------------------------------------|	
	""")

	st.markdown("")
	st.markdown("""If you have categories in your data that you're interested in,
		they can be included in your file names as **metadata**.
		Categories should be placed at the beginning of file names before an underscore:
		
		BIO_G0_02_1.txt, BIO_G0_03_1.txt, ENG_G0_16_1.txt, ENG_G0_21_1.txt, HIS_G0_02_1.txt, HIS_G0_03_1.txt
		""")
		
	st.markdown("""Processing these names would yield the categories:
		
		BIO, ENG, HIS
		""")
		
	st.markdown("""
		Those categories could then be compared in any combination.
		""")
		
	
	st.markdown("##### Adding a reference corpus (Optional)")
	st.markdown("""
	| :thumbsup: For most, it won't be necessary to process a reference corpus and metadata |
	|---------------------------------------------------------------------------------------|
	""")
	st.markdown("")
	
	st.markdown("""
		If you would like to make statistical comparisons (keywords), you can also create a reference corpus.
		Note that keywords can be generated using **metadata** as described above, as well.
		""")
	
	st.markdown("---")
	st.markdown("### Processing a target corpus (Required)")
	
	st.markdown("""
		Once your data have been prepared, use the **Manage Corpus Data** tool to process your data.
		There you will find a upload widget that will allow you to select your files.
		""")
	st.markdown("##### :books: Selecting a model (Required)")
	st.markdown("""
		You will also find a drop-down that will allow you to **select the model** you'd like to use.
		The model you select **will not affect part-of-speech** tags.
		It does change the **DocuScope** tagset. Each model was trained on data tagged with a different DocuScope dictionary.
		There are currently 2 to choose from:
		
		* The full dictionary. This has the most categories.
		* A "common dictionary". This has fewer categories with names less strictly rhetorical and more interdisciplinary.
		
		Your choice of model will depend on your goals and which tagset you find most explanatory.
		Are you trying to just build a predictive model? Or analyze patterns you need to explain to an audience?
		""")
	
	st.markdown("##### Processessing the data (Required)")
	st.markdown("""
	| :alarm_clock: It takes roughly 1 minute to process 1 million words|
	|---------------------------------------------------------------------|
	""")
	st.markdown("")
	st.markdown("""
		Once you've uploaded files into to the widget, you will see the file names.
		A **Process Corpus** button will also appear in the **lefthand sidebar**.
		When you're ready, clicking the button will send your data through the processing pipeline.
		The time it takes will depend on the size of the corpus.
		""")
	st.markdown("##### Processessing metadata and/or reference data (Optional)")
	st.markdown("""
		After you're taget corpus has been processed, other options will appear in the tool.
		You can choose to process additional data at any time.
		You will also see a button that will allow you **reset** your corpus.
		""")
	
	st.markdown("---")
	st.markdown("### :vertical_traffic_light: Navigating the tools")	
	st.markdown("""
		After processing your data, you can generate tables and plots showing various frequencies and relationships.
		Options and generation buttons will be located in the **lefthand sidebar**.
		""")
	st.markdown("##### Filtering tables")	
	st.markdown("""
		Once a table has been created, any of its columns can be filtered by clicking on **the hamburger menu** (the three lines) in a column header.
		""")
	st.markdown("##### Selecting rows")	
	st.markdown("""
		Specific rows can be selected by using the checkboxes that appear in a designated column.
		""")
	st.markdown("##### Downloading data")	
	st.markdown("""
		Plots can be saved by clicking on the **three dots** to the right of the plot.
		Tables can be downloaded by clicking the **Download** button in the sidebar.
		If you have used the checkboxes to select rows as described above, **only those rows** will be prepared for download.
		All tables are downloaded in an **Excel** format.
		""")
		
if __name__ == "__main__":
    main()