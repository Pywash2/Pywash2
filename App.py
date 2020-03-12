import dash

from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
import dash_table as dtb
import dash_daq as daq

app = dash.Dash(__name__, assets_folder='UI/assets')
server = app.server  # Reveal server to outside
app.config['suppress_callback_exceptions'] = True
app.title = 'PyWash'

#Create columntype-list
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



#This is the top-level interface, in here go all components
app.layout = html.Div(
    id = 'mainLayout',
    children = [
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
                    children=[html.A('Drag & drop csv or click here to get started')
                    ],
                    style = {'textAlign':'center','display': 'inline-block'}
                ),
                html.Div(
                    id = 'temp_button',
                    children = [
                        html.Button('Start(Temp)', id='button'),
                    ],
                    style = {'textAlign':'center','display': 'inline-block'}
                ),
            ],
            style = {'textAlign':'center','height': '100px'}
        ),

        #Stage 1

        html.Div(
            id = 'Stage_1',
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
                                        value='1'
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
                                        value='int64'
                                    ),
                                    style = {'width':'25%','display': 'inline-block','vertical-align': 'middle'}
                                )
                            ],
                            style = {'width':'60%','display': 'inline-block','vertical-align': 'middle'}
                        )
                        #could also put below 2 in 1 div and do width 40%,textalign center on div to center all ipv manual
                    ],
                    style = {'height': '100px','vertical-align': 'middle'}
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
                    style = {'height': '100px','vertical-align': 'middle'}
                ),
                html.Div(
                    id = 'temp_button2',
                    children = [
                        html.Button('Start', id='button2'),
                    ],
                    style = {'width': '100%','textAlign':'center','display': 'inline-block'}
                ),
                html.Div(
                    id = 'sample_data',
                    children = [
                        html.H5("Here comes some sample data"),
                        dtb.DataTable(
                            id="sample_data_table",
                            columns=TableInput,
                        )
                    ],
                    style = {'width': '30%','textAlign':'center'}
                )
            ],
#            style = {'display': 'none'}
        ),

        #Stage 2

        html.Div(
            id = 'Stage_2',
            children = [
                html.H3(
                    children = "Look at all of these not existing visualizations! Wow!",
                    style = {'textAlign':'center'}
                ),
            ],
            style = {'display': 'none'}
        )
    ]
)

@app.callback([Output('Data_Upload', 'style'),
              Output('Stage_1','style'),
              Output('Stage_2','style')],
              [Input('button', 'n_clicks'),
              Input('button2','n_clicks')])
def initiate_stages(click1,click2):
    if click2 != None:
        return {'visibility': 'hidden'},{'display': 'none'},{'display': 'block'}
    if click1 != None:
        return {'visibility': 'hidden'},{'display': 'block'},{'display': 'none'}

    return {'textAlign':'center','height': '100px','display': 'block'},{'display': 'none'},{'display': 'none'}

@app.callback(
    Output('dropdown_column_1', 'options'),
    [Input('dropdown_column_2', 'value')],
    [State('dropdown_column_1', 'value')],
)
def updateDropdown1(val2,val1):
    returnList = []
    i = 0
    for row in columnTypeList:
        i = i + 1
        if i is val1:
            row[1] = val2
        returnList.append({'label': row[0]+':'+row[1], 'value': i})
    return returnList

@app.callback(
    Output('dropdown_column_2', 'value'),
    [Input('dropdown_column_1', 'value')]
)
def updateDropdown2(val1):
    i = 0
    for row in columnTypeList:
        i = i + 1
        if i is val1:
            return row[1]
    return ''

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)
