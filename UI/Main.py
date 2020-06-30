import dash_html_components as html
import dash_core_components as dcc

from UI.DataCleaning import *
from UI.Visualization import *

def MainLayout():
    return html.Div(
        id = 'mainLayout',
        children = [
            dcc.Store(id='dataUploaded'),
            dcc.Store(id='dataProcessed'),
            #Title Section
            html.Div(
                id = 'titlePanel',
                children = [
                    html.H1(
                        children = "Pywash browser interface",
                        style = {'textAlign':'center'}
                    ),
                ]
            ),
            html.Div( #empty space
                style = {'height':'100px'},
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
                        style = {'textAlign':'center','display': 'inline-block','border':'3px dashed black','padding':'5px','font-weight':'bold'}
                    ),
                    html.Div( #empty space
                        style = {'height':'100px'},
                    ),
                    html.H3(
                        children = "Keep in mind that processing can take a while depending on the imported dataset",
                        style = {'textAlign':'center'}
                    ),
                ],
                style = {'textAlign':'center'}
            ),
#            dcc.Loading(
#                id = 'loadCleaning',
#                type="default",
#                children = [
            DataCleaningUI(),
#                ],
#            ),
            VisualizationUI(),
        ]
    )
