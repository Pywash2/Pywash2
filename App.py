import dash
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
import dash_table as dtb
import dash_daq as daq

from PyWash import *

app = dash.Dash(__name__, assets_folder='UI/assets')
server = app.server  # Reveal server to outside
#app.config['suppress_callback_exceptions'] = True
app.title = 'PyWash'

from UI.Main import *
from UI.Callbacks import *

app.layout = MainLayout()

#This is the top-level interface, in here go all components

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)
