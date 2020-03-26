from App import app
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
import dash_table

from dash.exceptions import PreventUpdate

from PyWash import SharedDataFrame

theData = None
OriginalData = None

# Temporary
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

# Load Data
@app.callback(Output('dataTrigger', 'data'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def store_data(data_contents, data_name, data_date):
    if data_contents is None:
        # prevent the None callbacks is important with the store component.
        # you don't want to update the store for nothing.
        raise PreventUpdate
    print("loading datasets: " + str(data_name))
    dataSet = SharedDataFrame(file_path=data_name, contents=data_contents, verbose=False)
    global theData
    global OriginalData
    theData = dataSet
    OriginalData = dataSet
    return 1
#    return dataSet

# Put Data in Table
@app.callback(Output('the_data_table', 'children'),
              [Input('dataTrigger', 'data')])
def data_table(change):
    df = theData.data
    return [html.Div([
        html.H5("The Data"),
        dcc.Store(id='memory-output'),
        dash_table.DataTable(
            id='datatable',
            columns=[
                {"name": i, "id": i, "deletable": True} for i in df.columns
            ],
            data=df.to_dict('records'),
#            editable=True,
#            filtering=True,
#            sorting=True,
#            sorting_type="multi",
#            row_selectable="multi",
#            row_deletable=True,
#            selected_rows=[],
#            pagination_mode="fe",
#            pagination_settings={
#                "displayed_pages": 1,
#                "current_page": 0,
#                "page_size": 50,
#            },
#            navigation="page",
#            style_cell={'textAlign': 'right', "padding": "5px"},
#            style_table={'overflowX': 'auto'},
#            style_cell_conditional=[{'if': {'row_index': 'odd'},
#                                     'backgroundColor': 'rgb(248, 248, 248)'}],
#            style_header={'backgroundColor': '#C2DFFF',
#                          'font-size': 'large',
#                          'text-align': 'center'},
#            style_filter={'backgroundColor': '#DCDCDC',
#                          'font-size': 'large'},
        ),
    ])]


# Main Callbacks
@app.callback([Output('Data_Upload', 'style'),
              Output('DataCleaning','style'),
              Output('Visualization','style')],
              [Input('button','n_clicks')])
def initiate_stages(click):
    if click != None:
        return {'visibility': 'hidden'},{'display': 'none'},{'display': 'block'}
    return {'textAlign':'center','height': '100px','display':'block'},{'display': 'block'},{'display': 'none'}

# Data Cleaning Callbacks
@app.callback(
    Output('dropdown_normalization', 'options'),
    [Input('dropdown_column_2', 'value')],
    [State('dropdown_normalization', 'value')],
)
def updateNormalizationColumns(val2,val1):
    returnList = []
    i = 0
    for row in columnTypeList:
        i = i + 1
        if i is val1:
            row[1] = val2
        returnList.append({'label': row[0]+':'+row[1], 'value': i})
    return returnList

@app.callback(
    Output('dropdown_column_1', 'options'),
    [Input('dropdown_column_2', 'value')],
    [State('dropdown_column_1', 'value')],
)
def updateColumnChooseNames(val2,val1):
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
    [Input('dropdown_column_1', 'value')],
)
def updateDColumnChooseValues(val1):
    i = 0
    for row in columnTypeList:
        i = i + 1
        if i is val1:
            return row[1]
    return ''
