import pathlib
from importlib.machinery import SourceFileLoader

# set paths
HERE = pathlib.Path(__file__).parents[1].resolve()
OPTIONS = str(HERE.joinpath("options.toml"))
IMPORTS = str(HERE.joinpath("utilities/handlers_imports.py"))

# import options
_imports = SourceFileLoader("handlers_imports", IMPORTS).load_module()
_options = _imports.import_options_general(OPTIONS)

if _options['global']['desktop_mode'] == False:
	from server.downloads import download
	from server.session import session_state

if _options['global']['desktop_mode'] == True:
	from ..server.downloads import download
	from ..server.session import session_state


