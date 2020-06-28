import dash
import os
import pkg_resources
import sys
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
    # Event Logger

    with open('requirements.txt', 'r') as file:
        packages = [i.split('==')[0] for i in file.readlines()]

    if os.path.exists('./eventlog.txt'):
        os.remove('./eventlog.txt')

    with open('eventlog.txt', 'w') as file:
        file.write('Versions of all main packages: \n')

        for pkg in packages:
            file.write(pkg + ' has version: ' + str(pkg_resources.get_distribution(pkg).version) + '\n')
        file.write('Python has version ' + str(sys.version.split(' ')[0]))
        file.write('\n\n\nLog of all functions: \n')

    # Run App

    app.run_server(debug=True, use_reloader=False)


