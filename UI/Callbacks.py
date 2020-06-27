from App import app

import dash
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

#Put Data in Preview Table
@app.callback(
    [Output('PreviewDataTable', 'columns'),
    Output('PreviewDataTable', 'style_cell_conditional'),
    Output('PreviewDataTable', 'data')],
    [Input('dataUploaded', 'data')],
)
def createDataPreview(change):
    print('trying to create data preview')
    df = previewData
    if df is not None:
        print('creating data preview')
        return([
            {"name": i, "id": i, "deletable": True, "renamable": True,} for i in df.columns
        ],
        create_conditional_style(df.columns),
        df.to_dict('records')
        )

def create_conditional_style(columns): #Fix headers of dataframe columns not fitting, taken from https://github.com/plotly/dash-table/issues/432\
    css=[]
    for col in columns:
        name_length = len(col)
        pixel = 50 + round(name_length*7)
        pixel = str(pixel) + "px"
        css.append({'if': {'column_id': col}, 'minWidth': pixel})
    print(css)
    return css


# Load Data
@app.callback(
    Output('dataUploaded', 'data'),
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename'),
    State('upload-data', 'last_modified')]
)
def store_data(data_contents, data_name, data_date):
    if data_contents == None:
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
    print("dataset loaded")
    return 1

#Put Data in Result Table
@app.callback(
    [Output('ResultDataTable', 'columns'),
    Output('ResultDataTable', 'style_cell_conditional'),
    Output('ResultDataTable', 'data')],
    [Input('dataProcessed','data')]
)
def createDataResult(change):
    if theData is not None:
        df = theData.get_dataframe()
        if df is not None:
            return ([
                {"name": i, "id": i, "deletable": True} for i in df.columns
            ],
            create_conditional_style(df.columns),
            df.to_dict('records')
            )
    return ([{"id": " ","name": " "}],[{}],[{}])

#Change data based on selected actions
@app.callback(
    Output('dataProcessed', 'data'),
    [Input('button','n_clicks')],
    [State('columnStorage', 'data'),
    State('missingValues', 'value'),
    State('dropdown_outliers', 'value'),
    State('dropdown_normalization', 'value'),
    State('dropdown_standardization', 'value'),
    State('DuplicatedRows','value')],
)
def processData(click, columnData, missingValues, handleOutliers, normalizationColumns, standardizationColumns, removeDuplicateRows):
    if click != None:
        print('Starting processing, this can take a while')
        print(columnData)
        theData.startCleaning(columnData, missingValues, handleOutliers, normalizationColumns, standardizationColumns, removeDuplicateRows)
        return 1
    return None

#Create column list based on dataTable
@app.callback(
    Output('columnStorage', 'data'),
    [Input('PreviewDataTable', 'columns'),
    Input('dropdown_column_2', 'value')],
    [State('dropdown_column_1', 'value'),
    State('columnStorage', 'data')],
)
def createColumnList(prevData,col2,col1,colData):
    #Look at what triggered the callback, 'PreviewDataTable' or 'dropdown_column_2'
    ctx = dash.callback_context
    last_event = ctx.triggered[0]['prop_id'].split('.')[0]

    #If already initialized, change column based on chosen type
    if colData != None:
        if last_event == 'dropdown_column_2':
            if col2 == '':
                #This means that the user hasn't chosen a datatype yet, so it shouldnt update
                raise PreventUpdate
            print('updating Column Storage')
            print(col1)
            print(col2)
            for row in colData:
                if row[0] == col1:
                    row[1] = str(col2)

            return colData
        #Update preprocessing data based on changes in preview table
        if last_event == 'PreviewDataTable':
            prevList = []
            i = 0
            for item in prevData:
                part2 = None
                for item2 in colData:
                    if item2[0] == item['name']: #Column deleted
                        part2 = item2[1]
                if part2 == None: #Column renamed
                    part2 = colData[i][1]
                #find matching string for item
                prevList.append([item['name'],part2])
                i += 1
            print(prevList)
            return prevList

    #Otherwise, initialize column list
    if last_event == 'PreviewDataTable' and prevData != None:
        dataFrame = previewData
        columnTypeList = list()
        columnList = list(dataFrame.columns)
        for item in columnList:
            x = [item,str(dataFrame[item].dtype)]
            print(x)
            columnTypeList.append(x)
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
    [Input('columnStorage', 'data')],
    [State('dropdown_column_1', 'value'),
    State('dropdown_column_2', 'value')],
)
def updateColumnChooseNames(colData,col1,col2):

    if colData != None:
        print('updating Column 1')
        print(col2)
        returnList = []
        for row in colData:
            print(row)
            returnList.append({'label': row[0]+':'+row[1], 'value': row[0]})
        return returnList
    return [{'label': 'Import data to get started', 'value': '0'}]

@app.callback(
    Output('dropdown_column_2', 'value'),
    [Input('dropdown_column_1', 'value')],
    [State('columnStorage', 'data')],
)
def updateDColumnChooseValues(col1,colData):
    if colData != None:
        print('updating Column 2')

        # Event Logger
        with open('event_logger.txt', 'a') as file:
            type_col = [i[1] for i in colData if i[0] == col1][0]
            string = 'The Data type of column ' + str(col1) + ' has been changed to ' + str(type_col) + '\n \n'
            file.write(string)

        i = 0
        for row in colData:
            i = i + 1
            if i == col1:
                return row[1]
            return ''

@app.callback(
    Output('dropdown_normalization', 'options'),
    [Input('columnStorage', 'data')],
    [State('dropdown_normalization', 'value')],
)
def updateNormalizationColumns(colData,norm):
    if colData != None:
        print('updating Normalization Column')
        returnList = []
        for row in colData:
            returnList.append({'label': row[0]+':'+row[1], 'value': row[0]})
        return returnList
    return [{'label': 'Import data to get started', 'value': '0'}]

@app.callback(
    Output('dropdown_standardization', 'options'),
    [Input('columnStorage', 'data')],
    [State('dropdown_standardization', 'value')],
)
def updateStandardizationColumns(colData,norm):
    if colData != None:
        print('updating Standardization Column')
        returnList = []
        for row in colData:
            returnList.append({'label': row[0]+':'+row[1], 'value': row[0]})
        return returnList
    return [{'label': 'Import data to get started', 'value': '0'}]

#Anomaly Callbacks
@app.callback(
    Output('dropdown_anomaly_1', 'options'),
    [Input('columnStorage','data'),
    Input('anomaliesbutton','n_clicks')],
    [State('dropdown_anomaly_1','options'),
    State('dropdown_anomaly_1','value')],
)
def input_anomaly_columnList(colData,click,options,value):
    ctx = dash.callback_context
    last_event = ctx.triggered[0]['prop_id'].split('.')[0]
    print('last event: ' + last_event)

    if colData != None and last_event == 'columnStorage':
        print('updating Anomaly Column List')
        returnList = []
        for key in theData.anomalies:
            returnList.append({'label': key, 'value': key})
        return returnList

    elif click != None and last_event == 'anomaliesbutton':
        print('deleting column from anomalylist')
        returnList = []
        for item in options:
            if item['value'] != value:
                returnList.append(item)
            else:
                theData.remove_anomaly_prediction(value)
        return returnList
    print('skipping anomalylist')
    return []


@app.callback(
    Output('dropdown_anomaly_2', 'options'),
    [Input('dropdown_anomaly_1','value')],
)
def input_anomaly_anomalyList(anomalyCol):
    print('editing anomaly list')
    if anomalyCol != '':
        returnList = []
        for item in theData.anomalies.get(anomalyCol):
            print(item)
            returnList.append({'label': item, 'value': item})
        return returnList
    return [{'label': 'Import data to get started', 'value': ''}]
