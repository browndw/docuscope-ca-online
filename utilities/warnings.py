
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

def warning_4(duplicates):
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

warning_5 = """
	<div style="background-color: #fddfd7; padding-left: 5px;">
	&#128555; Your data should contain at least 2 and no more than 20 categories. You can either proceed without assigning categories, or reset the corpus, fix your file names, and try again.
	</div>
	"""

warning_6 = """
	<div style="background-color: #fddfd7; padding-left: 5px;">
	&#128555; Your categories don't seem to be formatted correctly. You can either proceed without assigning categories, or reset the corpus, fix your file names, and try again.
	</div>
