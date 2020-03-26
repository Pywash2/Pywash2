import dash_html_components as html
import dash_core_components as dcc

import dash_daq as daq
import dash_table as dtb

#Example for column types
columnTypeList = list()
columnTypeList.append(['ID','int64'])
columnTypeList.append(['Name','object'])
columnTypeList.append(['Gender','bool'])
columnTypeList.append(['Weight','float64'])
columnTypeInput = []
TableInput = []
i = 0
for row in columnTypeList:
    i = i + 1
    columnTypeInput.append({'label': row[0]+':'+row[1], 'value': i})
    TableInput.append({'name': row[0], 'id': row[0]})

def DataCleaningUI():
    return html.Div(
        id = 'DataCleaning',
        children = [
        #Cleaning Options Layer 1: Column Type Detecting
            html.Div(
                id = 'Cleaning_Layer_1',
                children = [
                    html.Div(
                        html.H6("Cleaning Layer 1"),
                        style = {'width': '30%','display': 'inline-block','vertical-align': 'middle'}
                        ),
                    html.Div(
                        children = [
                            html.Div(
                                html.H6("If desired, change columns"),
                                style = {'width':'20%','display': 'inline-block','textAlign':'center','vertical-align': 'middle'}
                            ),
                            html.Div( #Check updateMapSelect in visualization for example of dynamically changing list entries
                                dcc.Dropdown(
                                    id = 'dropdown_column_1',
                                    options=columnTypeInput,
#                                    value='1'
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
                                        {'label': 'Category', 'value': 'category'},
                                        {'label': 'Error', 'value': 'error'}
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
                        html.H6("Cleaning Layer 2"),
                        style = {'width': '30%','display': 'inline-block'}
                    ), #could also put below 2 in 1 div and do width 40%,textalign center on div to center all ipv manual
                    html.Div(
                        id = 'Missing_Values',
                        children = [
                            html.H6("Test for missing values?"),
                            dcc.RadioItems(
                                options=[
                                    {'label': 'No', 'value': '0'},
                                    {'label': 'Yes', 'value': '1'},
                                ],
                                value='1',
                                labelStyle={'display': 'inline-block'}
                            )
                        ],
                        style = {'width': '20%','display': 'inline-block','textAlign':'center','vertical-align': 'middle'}
                    ),
                    html.Div(
                        id = 'Duplicated_Rows',
                        children = [
                            html.H6("Test for duplicated rows?"),
                            daq.BooleanSwitch(
                                id='Duplicated_Rows_Booleanswitch',
                                on=True
                            )
                        ],
                        style = {'width': '20%','display': 'inline-block','textAlign':'center','vertical-align': 'middle'}
                    )
                ],
                style = {'vertical-align': 'middle'}

            ),
            html.Div(
                id = 'Cleaning3',
                children = [
                    html.Div(
                        html.H6("Cleaning Layer 3"),
                        style = {'width': '30%','display': 'inline-block'}
                    ), #could also put below 2 in 1 div and do width 40%,textalign center on div to center all ipv manual
                    html.Div(
                        id = 'Outliers',
                        children = [
                            html.H6("Already test for outliers?"),
                            dcc.RadioItems(
                                options=[
                                    {'label': 'No', 'value': '0'},
                                    {'label': 'Yes', 'value': '1'},
                                ],
                                value='1',
                                labelStyle={'display': 'inline-block'}
                            )
                        ],
                        style = {'width': '20%','display': 'inline-block','textAlign':'center','vertical-align': 'middle'}
                    ),
                    html.Div(
                        id = 'Normalize',
                        children = [
                            html.H6("Already normalize column(s)?"),
                            dcc.Dropdown(
                                id = 'dropdown_normalization',
                                options=columnTypeInput,
                                multi=True,
                                value=""
                            )
                        ],
                        style = {'width': '30%','display': 'inline-block','textAlign':'left','vertical-align': 'middle'}
                    )
                ],
                style = {'vertical-align': 'middle'}
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
        ],
        style = {'textAlign':'center'}
    )
