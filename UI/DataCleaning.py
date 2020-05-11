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
            html.Div(
                id = 'Cleaning_Layer_1',
                children = [
                    html.Div(
                        html.H6("  "), #Creates a white space
                        style = {'width': '30%','display': 'inline-block','vertical-align': 'middle'}
                        ),
                    html.Div(
                        children = [
                            html.Div(
                                html.H6("If desired, change column types"),
                                style = {'width':'20%','display': 'inline-block','textAlign':'center','vertical-align': 'middle'}
                            ),
                            html.Div( #Check updateMapSelect in visualization for example of dynamically changing list entries
                                dcc.Dropdown(
                                    id = 'dropdown_column_1',
                                    options=[{'label': 'Import data to get started', 'value': '0'}],
                                    value="0",
                                ),
                                style = {'width':'40%','display': 'inline-block','vertical-align': 'middle'}
                            ),
                            html.Div(
                                dcc.Dropdown(
                                    id = 'dropdown_column_2',
                                    options=[
                                        {'label': 'Integer', 'value': 'int64'},
                                        {'label': 'Float', 'value': 'float64'},
                                        {'label': 'String', 'value': 'object'},
                                        {'label': 'Boolean', 'value': 'bool'},
                                        {'label': 'Date/Time', 'value': 'datetime64'},
                                        {'label': 'Categorical', 'value': 'category'},
                                    ],
#                                    value='int64'
                                ),
                                style = {'width':'25%','display': 'inline-block','vertical-align': 'middle'}
                            )
                        ],
                        style = {'width':'60%','display': 'inline-block','vertical-align': 'middle'}
                    )
                #could also put below 2 in 1 div and do width 40%,textalign center on div to center all ipv manual
                ],
                style = {'vertical-align': 'middle'}

            ),
        #Cleaning Options Layer 2: Missing Values & Duplicated Rows
            html.Div(
                id = 'Cleaning2',
                children = [
                    html.Div(
                        html.H6("   "), #Creates a white space
                        style = {'width': '40%','display': 'inline-block'}
                    ), #could also put below 2 in 1 div and do width 40%,textalign center on div to center all instead of manual
                    html.Div(
                        id = 'Missing_Values_Box',
                        children = [
                            html.H6("Test for missing values?"),
                            dcc.RadioItems(
                                id = 'missingValues',
                                options=[
                                    {'label': 'No', 'value': '0'},
                                    {'label': 'Yes', 'value': '1'},
                                ],
                                value='1',
                                labelStyle={'display': 'inline-block'}
                            )
                        ],
                        style = {'width': '20%','display': 'inline-block','vertical-align': 'middle'}
                    ),
                    html.Div(
                        id = 'Duplicated_Rows',
                        children = [
                            html.H6("Test for duplicated rows?"),
#                            daq.BooleanSwitch(
#                                id='Duplicated_Rows_Booleanswitch',
#                                on=True
#                            )
                            dcc.RadioItems(
                                id = 'DuplicatedRows',
                                options=[
                                    {'label': 'No', 'value': '0'},
                                    {'label': 'Yes', 'value': '1'},
                                ],
                                value='1',
                                labelStyle={'display': 'inline-block'}
                            )
                        ],
                        style = {'width': '20%','display': 'inline-block','vertical-align': 'middle'}
                    ),
                ],
                style = {'vertical-align': 'middle'}

            ),
            html.Div(
            id = 'outlier handling',
            children = [
                html.H6('Handle outliers?'),
                dcc.Dropdown(
                    id = 'dropdown_outliers',
                    options=[{'label': 'No', 'value': '0'},
                             {'label': 'Yes, mark in an extra column', 'value': '1'},
                             {'label': 'Yes, remove rows', 'value': '2'}],
                    multi=False,
                    value='1'
                ),
            ]
            ),
            html.Div(
            id = 'standardize/Normalize',
            children = [
                html.H6("Normalize column(s)?"),
                dcc.Dropdown(
                    id = 'dropdown_normalization',
                    options=[{'label': 'Import data to get started', 'value': '0'}],
                    multi=True,
                ),
                html.H6("Standardize column(s)?"),
                dcc.Dropdown(
                    id = 'dropdown_standardization',
                    options=[{'label': 'Import data to get started', 'value': '0'}],
                    multi=True,
                ),
            ]
            ),
            html.Div(
                id = 'emptySpace0',
                style = {'height':'40px'},
            ),
            html.Div(
                id = 'temp_button',
                children = [
                    html.Button('Start', id='button'),
                ],
                style = {'width': '100%','textAlign':'center','display': 'inline-block'}
            ),
            html.Div(
                id = 'preview_data',
                children = [
                    html.H5("Data Preview"),
                    dash_table.DataTable(
                        id='PreviewDataTable',
#                        columns=None,
#                        data=pd.DataFrame(),
                    ),
                ],
            )
        ],
        style = {'textAlign':'center'}
    )
