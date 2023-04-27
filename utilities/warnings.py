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

# Messages that that correspond to values stored in session state warnings.
# The default state (no warning) is 0.

warning_1 = """
	<div style="background-color: #fddfd7; padding-left: 5px;">
	&#128555; The files you selected could not be processed.
	Be sure that they are <b>plain text</b> files, that they are encoded as <b>UTF-8</b>, and that most of text is in English.
	For file preparation, we recommend that you use a plain text editor (and not an application like Word).
	</div>
	"""

def warning_2(duplicates):
    dups = ', '.join(duplicates)
    html_code = f'''
	<div style="background-color: #fddfd7; padding-left: 5px;">
	<p>&#128555; The files you selected could not be processed.
	Your corpus contains these <b>duplicate file names</b>:</p>
	<p><b>{dups}</b></p>
	Plese remove duplicates before processing.
	</div>
    '''
    return html_code

warning_3 = """
	<div style="background-color: #fddfd7; padding-left: 5px;">
	&#128555; Your corpus is too large for online processing.
	The online version of DocuScope Corpus Analysis & Concordancer accepts data up to roughly 3 million words.
	If you'd like to process more data, try <a href="https://github.com/browndw/docuscope-cac">the desktop version of the tool</a>, which available for free.
	</div>
	"""

def warning_4(exclusions):
	exclusions = ', '.join(exclusions)
	md = f"""##### The following documents were excluded from the corpus because they are improperly encoded:
	
	{exclusions}
	"""
	return(md)

def warning_5(duplicates):
    dups = ', '.join(duplicates)
    html_code = f'''
	<div style="background-color: #fddfd7; padding-left: 5px;">
	<p>&#128555; The files you selected could not be processed.
	Files with these <b>names</b> were also submitted as part of your target corpus:</p>
	<p><b>{dups}</b></p>
	Plese remove files from your reference corpus before processing.
	</div>
    '''
    return html_code

warning_6 = """
	<div style="background-color: #fddfd7; padding-left: 5px;">
	&#128555; The names in your target corpus and the names in the saved reference match.
	If these are different data, change the names and try again. Otherwise choose a different reference corpus.
	</div>
	"""

def warning_7(exclusions):
	exclusions = ', '.join(exclusions)
	md = f"""##### The following documents were excluded from the corpus because they match names in the reference:
	
	{exclusions}
	"""
	return(md)


warning_8 = """
	<div style="background-color: #fddfd7; padding-left: 5px;">
	&#128555; Your data should contain at least 2 and no more than 20 categories. You can either proceed without assigning categories, or reset the corpus, fix your file names, and try again.
	</div>
	"""

warning_9 = """
	<div style="background-color: #fddfd7; padding-left: 5px;">
	&#128555; Your categories don't seem to be formatted correctly. You can either proceed without assigning categories, or reset the corpus, fix your file names, and try again.
	</div>
	"""

warning_10 = """
	<div style="background-color: #fddfd7; padding-left: 5px;">
	&#128555; Your file name should include only letters, hyphens, or underscores.
	</div>
	"""

warning_11 = """
	<div style="background-color: #fddfd7; padding-left: 5px;">
	&#128528; It doesn't look like you've loaded a target corpus yet.
	</div>
	"""

warning_12 = """
	<div style="background-color: #fddfd7; padding-left: 5px;">
	&#127849; Your search didn't return any matches. Try something else.
	</div>
	"""

warning_13 = """
	<div style="background-color: #fddfd7; padding-left: 5px;">
	&#129327; Your search returned too many matches! Try something more specific.
	</div>
	"""

warning_14 = """
	<div style="background-color: #fddfd7; padding-left: 5px;">
	&#9999;&#65039; It doesn't look like you've entered a node word.
	</div>
	"""

warning_15 = """
	<div style="background-color: #fddfd7; padding-left: 5px;">
	&#128301; Your node word shouldn't contain any spaces.
	</div>
	"""

warning_16 = """
	<div style="background-color: #fddfd7; padding-left: 5px;">
	&#9986;&#65039; Your node word contains too many characters. Try something shorter.
	</div>
	"""

warning_17 = """
	<div style="background-color: #fddfd7; padding-left: 5px;">
	&#128528; It doesn't look like you've loaded a reference corpus yet.
	</div>
	"""
warning_18 = """
	<div style="background-color: #fddfd7; padding-left: 5px;">
	&#128200; Choose a variable to plot.
	</div>
	"""

warning_19 = """
	<div style="background-color: #fddfd7; padding-left: 5px;">
	&#128200; From the <b>Boxplots</b> menu, choose a variable to plot.
	</div>
	"""
warning_20 = """
	<div style="background-color: #fddfd7; padding-left: 5px;">
	&#128200; Choose at least one grouping variable from <b>Group A</b> and <b>Group B</b> to plot.
	</div>
	"""
