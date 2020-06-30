import dash_html_components as html
import dash_core_components as dcc
import dash_table

def VisualizationUI():
    return html.Div(
        id = 'Visualization',
        children = [
            html.H3(
                children = "Here you can have a quick look at the data before exporting to your computer",
                style = {'textAlign':'center'}
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
                            html.Button(
                                'Download cleaned dataset',
                                style = {'font-weight':'bold'},
                            ),
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
                html.H4("Dataset Summary"),
                style = {'width':'100%','display': 'inline-block','textAlign':'center','vertical-align': 'middle'}
            ),
            html.Div(
                dcc.Loading(
                    id = 'loadSummary',
                    type="default",
                    children = [
                        dash_table.DataTable(
                            id='summaryTable',
                        ),
                    ],
                ),
                style = {'width': '70%','display': 'block','vertical-align': 'middle', 'margin': 'auto'}
            ),
            html.Div( #empty space
                style = {'height':'50px'},
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
                                placeholder='Select column(s) for visualization',
                            ),
                            html.Div(
                                html.Button('Create visualization', id='visualizationbutton'),
                            ),
                        ],
                    ),
                    html.Div(
                        id = 'visualizationLocation',
                        children = [
                            dcc.Graph(
                                id = 'visGraph',
                                style = {
                                    'height':700
                                },
                                figure = { #Empty graph
                                    'data': [],
                                    'layout': {
                                        'title_text': '',
                                        'height': 700,  # px
                                    }
                                },
                            )
                        ],
                        style = {'width': '90%','display': 'block','vertical-align': 'middle', 'margin-left': 'auto', 'margin-right': 'auto'}
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
                                sort_action='native',
                                sort_mode='multi',
                                style_data_conditional=[
                                    {
                                        'if': {'row_index': 'odd'},
                                        'backgroundColor': 'rgb(248, 248, 248)'
                                    }
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
        style = {'display': 'none'}
    )
