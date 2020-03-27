from App import app
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
import dash_table

from dash.exceptions import PreventUpdate

from PyWash import SharedDataFrame

#These contain the imported data as SharedDataFrame objects
#originalData has the dataset as imported, previewData has the data shown when importing the dataset
theData = None
originalData = None
previewData = None
dataTypeList = [

]

# Load Data
@app.callback(
    Output('dataTrigger', 'data'),
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename'),
    State('upload-data', 'last_modified')]
)
def store_data(data_contents, data_name, data_date):
    if data_contents is None:
        # prevent the None callbacks is important with the store component.
        # you don't want to update the store for nothing.
        raise PreventUpdate
    print("loading datasets: " + str(data_name))
    dataSet = SharedDataFrame(file_path=data_name, contents=data_contents, verbose=False)
    global theData
    global OriginalData
    global previewData
    theData = dataSet
    OriginalData = dataSet
    previewData = dataSet.returnPreview()
    return 1
#    return dataSet

# Put Data in Table
@app.callback(
    Output('preview_data_table', 'children'),
    [Input('dataTrigger', 'data')]
)
def data_table(change):
    df = previewData
    return [html.Div([
        html.H5("Data Preview"),
        dash_table.DataTable(
            id='dataTable',
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

#Create column list based on dataTable
@app.callback(
    Output('columnStorage', 'data'),
    [Input('dataTable', 'data'),
    Input('dropdown_column_2', 'value')],
    [State('dropdown_column_1', 'value'),
    State('columnStorage', 'data')],

)
def createColumnList(dataChanges,col2,col1,colData):
    #If already initialized, change column based on chosen type
    if colData is not None:
        if col2 is '':
            #This means that the user hasn't chosen a datatype yet, so it shouldnt update
            raise PreventUpdate
        print('updating Column Storage')
        colData[col1] = [str(previewData.columns[col1]),str(col2)]
        return colData

    #Otherwise, initialize column list
    dataFrame = previewData
    columnTypeList = list()
    columnList = list(dataFrame.columns)
#    for item in columnList:
#        columnTypeList.append([item,str(dataFrame[item].dtype)])
    for item in columnList:
        columnTypeList.append([item,str(dataFrame[item].dtype)])
    return columnTypeList

# Main Callbacks
@app.callback(
    [Output('Data_Upload', 'style'),
    Output('DataCleaning','style'),
    Output('Visualization','style')],
    [Input('button','n_clicks')]
)
def initiate_stages(click):
    if click != None:
        return {'visibility': 'hidden'},{'display': 'none'},{'display': 'block'}
    return {'textAlign':'center','height': '100px','display':'block'},{'display': 'block'},{'display': 'none'}

# Data Cleaning Callbacks
@app.callback(
    Output('dropdown_column_1', 'options'),
#    Input('columnStorage', 'modified_timestamp'), #Hack, because empty storage creates issues
    [Input('columnStorage', 'data')],
    [State('dropdown_column_1', 'value'),
    State('dropdown_column_2', 'value')],
)
def updateColumnChooseNames(colData,col1,col2):
#    if ts is None:
#        raise PreventUpdate

    if colData is not None:
        print('updating Column 1')
        print(col2)
        returnList = []
        i = 0
        for row in colData:
            print(row)
            if i is col1:
                row[1] = col2
            returnList.append({'label': row[0]+':'+row[1], 'value': i})
            i = i + 1
        return returnList
    return [{'label': 'Import data to get started', 'value': '0'}]

@app.callback(
    Output('dropdown_column_2', 'value'),
    [Input('dropdown_column_1', 'value')],
    [State('columnStorage', 'data')],
)
def updateDColumnChooseValues(col1,colData):
    if colData is not None:
        print('updating Column 2')
        i = 0
        for row in colData:
            i = i + 1
            if i is col1:
                return row[1]
            return ''

@app.callback(
    Output('dropdown_normalization', 'options'),
    [Input('columnStorage', 'data')],
    [State('dropdown_column_2', 'value'),
    State('dropdown_normalization', 'value')],
)
def updateNormalizationColumns(colData,col2,norm):
    if colData is not None:
        print('updating Normalization Column')
        returnList = []
        i = 0
        for row in colData:
            i = i + 1
            if i is norm:
                row[1] = col2
            returnList.append({'label': row[0]+':'+row[1], 'value': i})
        return returnList
    return [{'label': 'Import data to get started', 'value': '0'}]
