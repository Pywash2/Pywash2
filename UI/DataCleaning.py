import dash_html_components as html
import dash_core_components as dcc
import dash_daq as daq
import dash_table

import pandas as pd

#Example for column types
def DataCleaningUI():
    return html.Div(
        id = 'DataCleaning',
        children = [
        #Cleaning Options Layer 1: Column Type Detecting
            dcc.Store(
                id = 'columnStorage',
            ),
            dcc.Store(
                id = 'anomalyBookkeeper',
            ),
            html.Div(
                id = 'Cleaning_Layer_1',
                children = [
                    html.Div(
                        html.H5("  "), #Creates a white space
                        style = {'width': '5%','display': 'inline-block','vertical-align': 'middle'}
                    ),
                    html.Div(
                        id = 'Column_Type_Changing',
                        children = [
                            html.Div(
                                html.H5("Check or change column types"),
                                style = {'width':'100%','display': 'inline-block','textAlign':'center','vertical-align': 'middle'}
                            ),
                            html.Div(
                                children = [
                                    html.Div( #Check updateMapSelect in visualization for example of dynamically changing list entries
                                        dcc.Dropdown(
                                            id = 'dropdown_column_1',
                                            options=[{'label': 'Import data to get started', 'value': '0'}],
                                            value="0",
                                            placeholder='Select to check or change column data type',
                                        ),
                                        style = {'width':'70%','display': 'inline-block','vertical-align': 'middle'}
                                    ),
                                    html.Div(
                                        dcc.Dropdown(
                                            id = 'dropdown_column_2',
                                            options=[
                                                {'label': 'Integer', 'value': 'int64'},
                                                {'label': 'Float', 'value': 'float64'},
                                                {'label': 'String', 'value': 'object'},
                                                {'label': 'Boolean', 'value': 'bool'},
                                                {'label': 'Date/Time', 'value': 'datetime64[ns]'},
                                                {'label': 'Categorical', 'value': 'category'},
                                            ],
                                            placeholder=' ',
                                        ),
                                        style = {'width':'30%','display': 'inline-block','vertical-align': 'middle'}
                                    ),
                                ],
                            ),
                        ],
                        style = {'width':'40%','display': 'inline-block','vertical-align': 'middle'}
                    ),
                    html.Div(
                        html.H5("  "), #Creates a white space
                        style = {'width': '10%','display': 'inline-block','vertical-align': 'middle'}
                    ),
                    html.Div(
                        id = 'Anomaly_Checking',
                        children = [
                            html.Div(
                                html.H5("Inspect found anomalies per column, change column type or delete columns that are not anomalous"),
                                style = {'width':'100%','display': 'inline-block','textAlign':'center','vertical-align': 'middle'}
                            ),
                            html.Div(
                                children = [
                                    html.Div(
                                        dcc.Dropdown(
                                            id = 'dropdown_anomaly_1',
                                            placeholder='Check which columns have anomalies',
                                        ),
                                        style = {'width': '50%','display': 'inline-block','vertical-align': 'middle'}
                                    ),
                                    html.Div(
                                        dcc.Dropdown(
                                            id = 'dropdown_anomaly_2',
                                            multi = True,
                                            placeholder='Select possible anomalies',
                                        ),
                                        style = {'width': '30%','display': 'inline-block','vertical-align': 'middle'}
                                    ),
                                    html.Div(
                                        html.Button('Select All', id='anomaliesButtonSelectAll'),
                                        style = {'width': '20%','display': 'inline-block','vertical-align': 'middle'}
                                    )
                                ],
                                style = {'width': '100%','display': 'inline-block','vertical-align': 'middle'}
                            ),
                            html.Div(
                                children = [
                                    html.Button('Selected items are not anomalies', id='anomaliesButtonNotAnomalies'),
                                    html.Button('Selected items are anomalies, handle them', id='anomaliesButtonYesAnomalies'),
                                ],
                            )
                        ],
                        style = {'width':'40%','display': 'inline-block','vertical-align': 'middle'}
                    ),
                    html.Div(
                        html.H5("  "), #Creates a white space
                        style = {'width': '5%','display': 'inline-block','vertical-align': 'middle'}
                    ),
                ],
                style = {'vertical-align': 'middle'}
            ),
            html.Div( #empty space
                style = {'height':'50px'},
            ),
        #Cleaning Options Layer 2: Outlier Handling, Duplicated Rows, Missing Values
            html.Div(
                id = 'Cleaning_Layer_2',
                children = [
                    html.Div(
                        html.H5("   "), #Creates a white space
                        style = {'width': '5%','display': 'inline-block'}
                    ), #could also put below 2 in 1 div and do width 40%,textalign center on div to center all instead of manual
                    html.Div(
                        id = 'outlier handling',
                        children = [
                            html.Div(
                                html.H5('Choose preferred method for handling outliers'),
                            ),
                            html.Div(
                                dcc.Dropdown(
                                    id = 'dropdown_outliers',
                                    options=[
                                        {'label': 'Do not handle outliers', 'value': '0'},
                                        {'label': 'Slow & Precise: Mark in an extra column', 'value': '1'},
                                        {'label': 'Slow & Precise: Remove rows', 'value': '2'},
                                        {'label': 'Quick & Sloppy: Mark in an extra column', 'value': '3'},
                                        {'label': 'Quick & Sloppy: Remove rows', 'value': '4'},
                                    ],
                                    multi=False,
                                    value='1'
                                ),
                            ),
                        ],
                        style = {'width': '35%','display': 'inline-block','vertical-align': 'middle'}
                    ),
                    html.Div(
                        html.H5("   "), #Creates a white space
                        style = {'width': '5%','display': 'inline-block'}
                    ), #could also put below 2 in 1 div and do width 40%,textalign center on div to center all instead of manual
                    html.Div(
                        id = 'Missing_Values_Box',
                        children = [
                            html.Div(
                                html.H5("Select column to clean missing values, only for regression or classification target variable"),
                            ),
                            html.Div(
                                dcc.Dropdown(
                                    id = 'dropdown_missingValues',
                                    options=[],
                                    style={'display': 'inline-block','width': '100%'},
                                    placeholder='Select desired columns for missing value detection',
                                )
                            )
                        ],
                        style = {'width': '35%','display': 'inline-block','vertical-align': 'middle'}
                    ),
                    html.Div(
                        html.H5("   "), #Creates a white space
                        style = {'width': '5%','display': 'inline-block'}
                    ),
                    html.Div(
                        id = 'Duplicated_Rows',
                        children = [
                            html.Div(
                                html.H5("Test for duplicated rows?"),
                            ),
                            html.Div(
                                dcc.RadioItems(
                                    id = 'DuplicatedRows',
                                    options=[
                                        {'label': 'No', 'value': '0'},
                                        {'label': 'Yes', 'value': '1'},
                                    ],
                                    value='1',
                                    labelStyle={'display': 'inline-block'}
                                )
                            )
                        ],
                        style = {'width': '10%','display': 'inline-block','vertical-align': 'middle'}
                    ),
                    html.Div(
                        html.H5("   "), #Creates a white space
                        style = {'width': '5%','display': 'inline-block'}
                    ),
                ],
                style = {'vertical-align': 'middle'}

            ),
            html.Div( #empty space
                style = {'height':'50px'},
            ),
            #Standardize & Normalize
            html.Div(
                id = 'Cleaning_Layer_3',
                children = [
                    html.Div(
                        html.H5("  "), #Creates a white space
                        style = {'width': '5%','display': 'inline-block','vertical-align': 'middle'}
                    ),
                    #Normalize
                    html.Div(
                        id = 'normalization',
                        children = [
                            html.Div(
                                html.H5("Normalize column(s)?"),
                                style = {'width': '80%','display': 'inline-block','vertical-align': 'middle',}
                            ),
                            html.Div(
                                dcc.Dropdown(
                                    id = 'dropdown_normalization',
                                    options=[{'label': 'Import data to get started', 'value': '0'}],
                                    multi=True,
                                    placeholder='Select desired columns for normalization',
                                ),
                            ),
                        ],
                        style = {'width':'40%','display': 'inline-block','vertical-align': 'middle'}
                    ),
                    html.Div(
                        html.H5("  "), #Creates a white space
                        style = {'width': '10%','display': 'inline-block','vertical-align': 'middle'}
                    ),
                    #Standardize
                    html.Div(
                        id = 'standardization',
                        children = [
                            html.Div(
                                html.H5("Standardize column(s)?"),
                                style = {'width': '80%','display': 'inline-block','vertical-align': 'middle',}
                            ),
                            html.Div(
                                dcc.Dropdown(
                                    id = 'dropdown_standardization',
                                    options=[{'label': 'Import data to get started', 'value': '0'}],
                                    multi=True,
                                    placeholder='Select desired columns for standardization',
                                ),
                            ),
                        ],
                        style = {'width':'40%','display': 'inline-block','vertical-align': 'middle'}
                    ),

                    html.Div(
                        html.H5("  "), #Creates a white space
                        style = {'width': '5%','display': 'inline-block','vertical-align': 'middle'}
                    ),
                ],
                style = {'vertical-align': 'middle'}
            ),
            html.Div( #empty space between options and start/preview
                style = {'height':'50px'},
            ),
            html.Div(
                id = 'start_button',
                children = [
                    html.Button('Start Data Processing', id='startButton',style = {'font-weight':'bold'}),
                ],
                style = {'width': '100%','textAlign':'center','display': 'inline-block'}
            ),
            html.Div(
                id = 'preview_data',
                children = [
                    dcc.Loading(
                        id = 'loadPreview',
                        type="default",
                        children = [
                            html.H3("Data Preview"),
                            dash_table.DataTable(
                                id='PreviewDataTable',
                            ),
                        ],
                    ),
                ],
            )
        ],
        style = {'textAlign':'center'}
    )
