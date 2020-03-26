import dash_html_components as html
import dash_core_components as dcc

from UI.DataCleaning import *
from UI.Visualization import *

def MainLayout():
    return html.Div(
        id = 'mainLayout',
        children = [
            dcc.Store(id='dataTrigger'),
            #Title Section
            html.Div(
                id = 'titlePanel',
                children = [
                    html.H2(
                        children = "Welcome to the Pywash browser interface",
                        style = {'textAlign':'center'}
                    ),
                    html.H4(
                        children = "(In development)",
                        style = {'textAlign':'center'}
                    ),
                ]
            ),
            #Load Data Section
            html.Div(
                id = 'Data_Upload',
                children = [
                    dcc.Upload(
                        id='upload-data',
                        children=[
                            html.A('Drag & drop csv or click here to get started')
                        ],
                        multiple = False,
                        style = {'textAlign':'center','display': 'inline-block'}
                    ),
                ],
                style = {'textAlign':'center'}
            ),

            DataCleaningUI(),

            VisualizationUI(),

            html.Div(
                id = 'the_data_table',
                children = [
                ],
#                style = {'width': '50%','textAlign':'center','display': 'inline-block'}
            )

        ]
    )
