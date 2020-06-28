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
                                }
                            )
                        ],
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
