import dash_html_components as html
import dash_core_components as dcc
import dash_table

def VisualizationUI():
    return html.Div(
        id = 'Visualization',
        children = [
            html.H3(
                children = "Now, let's have a closer look!",
                style = {'textAlign':'center'}
            ),
            html.H5(
                children = "(Processing can take a while, it is done when the preview data loads at the bottom of the page)",
                style = {'textAlign':'center'}
            ),
            html.Div(
                id = 'Visualizing',
                children = [
                    html.Div(
                        id = 'visualization_selector',
                        children = [
                            html.H6("Select the primary column for visualization"),
                            dcc.Dropdown(
                                id = 'visualization_dropdown',
                                options = [],
                                value = '',
                                multi=True,
                            ),
                            html.Div(
                                [
                                    html.Button('Create visualization', id='visualizationbutton'),
                                    html.Button('Create data summary', id='summarybutton'),
                                ],
                            ),
                        ],
                    ),
                    html.Div(
                        id = 'visualizationLocation',
                        children = [
                            dcc.Graph(
                                id = 'visGraph',
                                style = {
                                    'height':500
                                },
                                figure = { #Empty graph
                                    'data': [],
                                    "layout": {
                                        "title": "My Dash Graph",
                                        "height": 700,  # px
                                    }
                                },
                            )
                        ],
                    ),
                ]
            ),
            html.Div(
                id = 'data_export',
                children = [
                    html.Div(
                        html.H5("  "), #Creates a white space
                        style = {'width': '45%','display': 'inline-block','vertical-align': 'middle'}
                    ),
                    html.Div(
                        html.A(
                            html.Button('Download cleaned dataset'),
                            id='downloadButton',
                            download = '',
                            href='',
                        ),
                        style = {'width': '10%','display': 'inline-block','vertical-align': 'middle'}
                    ),
                    html.Div(
                        html.H5("  "), #Creates a white space
                        style = {'width': '1%','display': 'inline-block','vertical-align': 'middle'}
                    ),
                    html.Div(
                        dcc.Dropdown(
                            id='downloadType',
                            options=[
                                {'label': '.CSV', 'value': 'csv'},
                                {'label': '.ARFF', 'value': 'arff'},
                            ],
                            value='csv',
                            multi=False,
                            placeholder='Select export file type',
                        ),
#                        style = {'width': '20%','display': 'inline-block'}
                        style = {'width': '8%','display': 'inline-block','vertical-align': 'middle'}
                    ),
                    html.Div(
                        html.H5("  "), #Creates a white space
                        style = {'width': '36%','display': 'inline-block','vertical-align': 'middle'}
                    ),
                ]
            ),
            html.Div(
                id = 'result_data',
                children = [
                    html.Div(
                        [
                            html.H5("Data Preview"),
                            dash_table.DataTable(
                                id='ResultDataTable',
                            ),
                        ],
                    ),
                ],
            ),
        ],
        style = {'display': 'none'}
    )
