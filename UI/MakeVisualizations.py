import pandas as pd
import numpy as np
import random

from wordcloud import WordCloud
import matplotlib.pyplot as plt

from plotly import tools as tls

import plotly.graph_objects as go
import plotly.express as px

def chooseVisualization(df,columns):
    print('starting visualizations')
    foundVis = None
    colAmount = len(columns)
    #First, set actual columns used and their types
    if colAmount == 0:
        return None

    else: #Try to visualize
        cols = []
        colType = []
        i = 0
        for item in columns:
            colType.append(typeDetector(df[columns[i]]))
            print('column ' + str(i) + ' type: ' + str(colType[i]))
            i = i + 1

    #Next, we check which visualization is correct
    if colAmount == 1: #1-column visualizations
        print('One Column')
        if colType[0] == 'category':
            #Plot: Bar Chart
            print('1-column, categorical')
            foundVis = '1-column Bar Chart'
        elif colType[0] == 'numeric' or colType[0] == 'date/time':
            #Plot: Histogram
            print('1-column, numerical')
            foundVis = 'Histogram'

    if colAmount > 1 and all(x == 'category' for x in colType):
        #Plot: Stacked Bar Chart
        print('All-categorical')
        foundVis = 'Multi-column Stacked Bar Chart'

    if colAmount == 2: #2-column visualizations
        print('Two Columns')
        if any(x == 'category' for x in colType) and any(x == 'numeric' for x in colType):
            #Plot: Scatterplot
            print('2-column, categorical/numerical')
            foundVis = 'Categorical Scatterplot'
        if all(x == 'numeric' for x in colType):
            #Plot: 'Normal' 2D Plot
            print('2-column, numerical/numerical')
            foundVis = 'Numerical Scatterplot'

    if foundVis == None:
        print('Error: could not find appropriate ' + str(colAmount) + '-column visualization')
        return 'No possible plot could be found'
    return foundVis

def typeDetector(col):
    colType = str(col.dtype)
    print(colType)
    if colType == 'category' or colType == 'bool':
        return 'category'
    elif colType == 'object':
        return 'text'
    elif colType == 'int64' or colType == 'float64' or colType == 'datetime64[ns]': #Numerical
        return 'numeric'
    else:
        print('Error! Actual type: ' + colType)
        return 'Error: Type Not Found'

def createVisualization(data,chosenVis):
    print("Trying to make visualization: '"+chosenVis+"'")
    if chosenVis == '1-column Bar Chart':
        return CreateBarChart(data)
    if chosenVis == 'Histogram':
        return createHistogram(data)
    if chosenVis == 'Multi-column Stacked Bar Chart':
        return createStackedBarChart(data)
    if chosenVis == 'Categorical Scatterplot':
        return createScatterPlot(data)
    if chosenVis == 'Numerical Scatterplot':
        return createTwoDPlot(data)

def CreateBarChart(data):
    print('creating 1-column Bar Chart')
    data = data.iloc[:,0]
    #Create list of unique categories + amount, then create figure
    uniqueGrouped = data.value_counts().sort_values(ascending=False)
    fig = px.bar(uniqueGrouped, x=uniqueGrouped.index, y=uniqueGrouped.tolist())
    return fig

def createHistogram(data):
    #Distribution Plot
    ###TODO: Implement nbins
    fig = px.histogram(data)
    return fig

def createStackedBarChart(data):
    #Partly from the previous year
    print('trying value counts apply')
    newDF = []
    for item in data.columns:
        column = data[item]
        column = column.value_counts()
        column = column.to_frame()
        column.index = column.index.map(str)
        newDF.append(column)
    totalDF = newDF[0]
    for i in range(1,len(newDF)):
        totalDF = pd.concat([totalDF,newDF[i]], axis=1)
    #Turn list of series into DF
    fig = go.Figure()
    for i in range(len(totalDF)):
        fig.add_trace(
            go.Bar(
                x = totalDF.columns,
                y = totalDF.values[i],
                name= str(totalDF.index[i]) #NOT X LABEL, ITEM LABEL
            )
        )
    fig.update_layout(
        barmode='stack',
    )
    return fig

def createScatterPlot(data):
    #Because it has categorical data, add jitter to get extra info
    col0T = typeDetector(data[data.columns[0]])
    if col0T == 'category':
        xCat = data.columns[0]
        x = data[data.columns[0]]
        yCat = data.columns[1]
        y = data[data.columns[1]]
    else:
        xCat = data.columns[1]
        x = data[data.columns[1]]
        yCat = data.columns[0]
        y = data[data.columns[0]]
    d = []
    for i in range(len(pd.unique(data[xCat]))):
        col = {
            'name': str(pd.unique(data[xCat])[i]),
            'type': 'violin',
            'x': data[xCat][data[xCat] == pd.unique(data[xCat])[i]],
            'y': data[yCat][data[xCat] == pd.unique(data[xCat])[i]],
            'points': 'all',
            'jitter': 1.0,
            'pointpos': 0,
            'line': {
                'width': 0
            },
            'fillcolor': 'rgba(0,0,0,0)',
        }
        d.append(col)

    fig = go.Figure(
        {
            'data': d,
            'layout': {
                'title': '',
            }
        }
    )
    return fig

def createTwoDPlot(data):
    fig = px.scatter(data, x = data[data.columns[0]], y = data[data.columns[1]])
    return fig
