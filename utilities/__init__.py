import pathlib
import tomli

HERE = pathlib.Path(__file__).parents[1].resolve()
OPTIONS = str(HERE.joinpath("options.toml"))

# import options
with open(OPTIONS, mode="rb") as fp:
	_options = tomli.load(fp)

DESKTOP = _options['global']['desktop_mode']

if DESKTOP == False:
	from server.downloads import download
	from server.session import session_state

if DESKTOP == True:
	from ..server.downloads import download
	from ..server.session import session_state


