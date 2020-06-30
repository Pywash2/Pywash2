from App import app

import dash
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
import dash_table
from dash.exceptions import PreventUpdate

from PyWash import SharedDataFrame
from UI.MakeVisualizations import *

from datetime import datetime

#These contain the imported data as SharedDataFrame objects
#originalData has the dataset as imported, previewData has the data shown when importing the dataset
theData = None
originalData = None
previewData = None

anomaliesDelete = []
anomaliesReplace = []
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

### Load Data
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
        df = theData.data
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
    [Input('startButton','n_clicks')],
    [State('columnStorage', 'data'),
    State('dropdown_missingValues', 'value'),
    State('dropdown_outliers', 'value'),
    State('dropdown_normalization', 'value'),
    State('dropdown_standardization', 'value'),
    State('DuplicatedRows','value')],
)
def processData(click, columnData, missingValues, handleOutliers, normalizationColumns, standardizationColumns, removeDuplicateRows):
    if click != None and theData != None:
        print('Starting processing, this can take a while')
        #Set column types (to be changed)
        colTypes = {}
        for item in columnData:
            colTypes[item[0]] = item[1]
        theData.col_types = colTypes
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
        types = theData.col_types
        for item in columnList:
            x = [item,str(types[item])]
            print(x)
            columnTypeList.append(x)
        return columnTypeList

### Data Cleaning Callbacks
@app.callback(
    Output('dropdown_column_1', 'options'),
    [Input('columnStorage', 'data')],
    [State('dropdown_column_1', 'value'),
    State('dropdown_column_2', 'value')],
)
def updateColumnChooseNames(colData,col1,col2):

    if colData != None:
        print('updating Column 1')

        if col2 != None:
            # Event Logger
            with open('eventlog.txt', 'a') as file:

                string = 'The Data type of column ' + str(col1) + ' has been changed to ' + str(col2) + '\n \n'
                file.write(string)

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
        print('updating Column 2: ' + str(col1))
        print(colData)
        for row in colData:
            print(row)
            if row[0] == col1:
                return row[1]
        return ''

@app.callback(
    [Output('dropdown_missingValues', 'options'),
    Output('dropdown_normalization', 'options'),
    Output('dropdown_standardization', 'options')],
    [Input('columnStorage', 'data')],
    [State('dropdown_normalization', 'value')],
)
def updateOtherColumns(colData,norm):
    if colData != None:
        print('updating Normalization Column')
        returnList = []
        for row in colData:
            returnList.append({'label': row[0]+':'+row[1], 'value': row[0]})
        return [returnList,returnList,returnList]
    returnItem = [{'label': 'Import data to get started', 'value': '0'}]
    return [returnItem,returnItem,returnItem]

### Anomaly Callbacks
@app.callback(
    [Output('dropdown_anomaly_1', 'options'),
    Output('anomalyBookkeeper','data')],
    [Input('columnStorage','data'),
    Input('anomaliesButtonNotAnomalies','n_clicks'),
    Input('anomaliesButtonYesAnomalies','n_clicks')],
    [State('dropdown_anomaly_1','options'),
    State('dropdown_anomaly_2','value'),
    State('dropdown_anomaly_1','value'),
    State('anomalyBookkeeper','data')],
)
def handleAnomalies(colData,notAnomalies,replaceAnomalies,coloptions,itemvalues,colvalue,bookKeeper):
    ctx = dash.callback_context
    last_event = ctx.triggered[0]['prop_id'].split('.')[0]
    if colData != None:
        if notAnomalies != None and last_event == 'anomaliesButtonNotAnomalies' and bookKeeper != None:

            with open('eventlog.txt', 'a') as file:
                valstring = ''
                for item in itemvalues:
                    valstring = valstring + str(item) + ','
                string = 'From column: ' + str(colvalue) + ', the following anomalies: ' + valstring + ' have been unmarked as anomalies.' + '\n' + '\n'
                file.write(string)

            print('Not anomalies: deleting item(s) from anomalylist')
            theData.remove_anomaly_prediction(colvalue,itemvalues)
            returnList = []
            for key in theData.anomalies:
                returnList.append({'label': key, 'value': key})
                bookKeeperUpdate = datetime.now()
            return [returnList,bookKeeperUpdate]
        if replaceAnomalies != None and last_event == 'anomaliesButtonYesAnomalies' and bookKeeper != None:
            print('Anomalies: replacing item(s) with None')

            with open('eventlog.txt', 'a') as file:
                valstring = ''
                for item in itemvalues:
                    valstring = valstring + str(item) + ','
                string = 'From column: ' + str(colvalue) + ', the following anomalies: ' + valstring + ' have been set to missing value annotator' + '\n' + '\n'
                file.write(string)

            theData.replace_anomalies(colvalue,itemvalues)
            returnList = []
            for key in theData.anomalies:
                returnList.append({'label': key, 'value': key})
            return [returnList,bookKeeper]
        if last_event == 'columnStorage':
            returnList = []
            for key in theData.anomalies:
                returnList.append({'label': key, 'value': key})
            return [returnList,bookKeeper]
    else:
        print('skipping anomalylist')
        return [[],bookKeeper]

@app.callback(
    Output('dropdown_anomaly_2','options'),
    [Input('dropdown_anomaly_1','value'),
    Input('anomalyBookkeeper','data')],
)
def updateAnomaliesListOptions(colValue,bookKeeper):
    print('editing anomaly list')
    if colValue != None:
        returnList = []
        for item in theData.anomalies.get(colValue):
            print(item)
            returnList.append({'label': item, 'value': item})
        return returnList
    return []

@app.callback(
    Output('dropdown_anomaly_2','value'),
    [Input('dropdown_anomaly_2','options'),
    Input('anomaliesButtonSelectAll','n_clicks'),
    Input('anomaliesButtonNotAnomalies','n_clicks')],
    [State('dropdown_anomaly_2','options'),
    State('dropdown_anomaly_2','value')],
)
def refreshAnomaliesListValue(optionsChanged,clickedSelectAll,clickedNotAnomalies,options,value):
    ctx = dash.callback_context
    last_event = ctx.triggered[0]['prop_id'].split('.')[0]

    if last_event == 'dropdown_anomaly_2' or last_event == 'anomaliesButtonNotAnomalies':
        return []
    if clickedSelectAll != None and last_event == 'anomaliesButtonSelectAll':
        print('selecting all anomalies')
        returnList = []
        for item in options:
            returnList.append(item['value'])
        return returnList

### Visualization callbacks

@app.callback(
    [Output('summaryTable', 'columns'),
    Output('summaryTable', 'style_cell_conditional'),
    Output('summaryTable', 'data')],
    [Input('dataProcessed','data')]
)
def give_summary(dataProcessed):
    if theData is not None:
        #Create summary
        dfList = []
        for item in theData.data.columns:
            df = pd.DataFrame({item: theData.data[item].describe()})
            dfList.append(df)
        totalDf = dfList[0]
        print(totalDf)
        if len(dfList) > 0:
            for i in range(1,len(dfList)):
                totalDf = pd.concat([totalDf, dfList[i]], axis=1)
                print(totalDf)
        totalDf = totalDf.reset_index()
        totalDf.rename(columns={'index':' '}, inplace=True)
        #Return summary to Dash
        return (
            [{"name": i, "id": i} for i in totalDf.columns],
            create_conditional_style(totalDf.columns),
            totalDf.to_dict('records')
        )
    return ([{"id": " ","name": " "}],[{}],[{}])

@app.callback(
    Output('visualization_dropdown','options'),
    [Input('ResultDataTable', 'data')],
)
def input_visualization_columnLists(dataProcessed):
#If style of visualization has been changed to visible, data has been processed, so lists can be populated
    if theData is not None:
        data = list(theData.data.columns.values)
        returnList = []
        for item in data:
            print(item)
            returnList.append({'label': item, 'value': item})
        print('ready for visualizations')
        return returnList
    return [{'label':'Please wait...','value':'0'}]
#    return [,]

@app.callback(
    Output('visualizationbutton','children'),
    [Input('visualization_dropdown','value')],
)
def updatePossibleVisualizations(columns):
    if columns != '':
        foundVis = chooseVisualization(theData.data,columns)
        return foundVis
    return 'Select column(s) for visualization'

@app.callback(
    Output('visGraph','figure'),
    [Input('visualizationbutton','n_clicks')],
    [State('visualizationbutton','children'),
    State('visualization_dropdown','value')],
)
def show_visualization(click,visName,columns):
    print('creating visualization...')
    if theData is not None:
        chosenVis = createVisualization(theData.data[columns],visName)
        return chosenVis
    return { #Empty graph
        'data': [],
        "layout": {
            "title": "",
            "height": 700,  # px
        }
    }

### Output callbacks
@app.callback(
    [Output('downloadButton', 'href'),
    Output('downloadButton', 'download')],
    [Input('dataProcessed', 'data'),
    Input('downloadType', 'value')],
)
def update_download_link(dataProcessed, downloadType):
    if dataProcessed == 1:
        fileString = 'cleaned' + theData.name + '.' + downloadType
        print(fileString)
        return[theData.export_string(downloadType),fileString]
    return ['','']

### Main Callbacks
@app.callback(
    [Output('Data_Upload', 'style'),
    Output('DataCleaning','style'),
    Output('Visualization','style')],
    [Input('startButton','n_clicks')]
)
def initiate_stages(click):
    if click != None and theData != None:
        return {'visibility': 'hidden'},{'display': 'none'},{'display': 'block'}
    return {'textAlign':'center','height': '100px','display':'block'},{'display': 'block'},{'display': 'none'}
