import pandas as pd

from wordcloud import WordCloud
from plotly import tools as tls
import matplotlib.pyplot as plt

import plotly.express as px

def chooseVisualization(df,columns):
    print('starting visualizations')
    foundVis = None
    colAmount = len(columns)
    #First, set actual columns used and their types
    if colAmount == 0:
        return 'Select columns for visualization'

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
# Wordcloud currently disabled, because i tried, and it didn't work
#        elif colType[0] == 'text':
#            #Plot: Word Cloud
#            print('1-column, text')
#            foundVis = 'Word Cloud'
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
            foundVis = 'Scatterplot'
        if all(x == 'numeric' for x in colType):
            #Plot: 'Normal' 2D Plot
            print('2-column, numerical/numerical')
            foundVis = '2D line plot'
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

def dateTimeTransformer(col):
    return None

def createVisualization(data,chosenVis):
    print("Trying to make visualization: '"+chosenVis+"'")
    if chosenVis == '1-column Bar Chart':
        return CreateBarChart(data)
    if chosenVis == 'Histogram':
        return createHistogram(data)
    if chosenVis == 'Multi-column Stacked Bar Chart':
        return createStackedBarChart(data)
    if chosenVis == 'Scatterplot':
        return createScatterPlot(data)
    if chosenVis == '2D line plot':
        return createTwoDPlot(data)

def CreateBarChart(data):
    print('creating 1-column Bar Chart')
    print(data.head(5))
    data = data.iloc[:,0]
    #Create list of unique categories + amount, then create figure
    uniqueGrouped = data.groupby(data).size().sort_values(ascending=False)
    fig = px.bar(uniqueGrouped, x=uniqueGrouped.index, y=uniqueGrouped.tolist())
    return fig

def createWordCloud(data):
    #Separate the data into words
    wordList = []
    for item in data:
        itemList = str(item).split()
        #Separate words
        for word in itemList:
#            realWord = re.sub("[^a-zA-Z]","",str(word))
#            if isinstance(word,str):
            wordList.append(str(word))
    giantString = " ".join(wordList)
    wordcloud = WordCloud().generate(giantString)
    fig = plt.figure()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plot = tls.mpl_to_plotly(fig) #Plotly.tools function
    return plot

def createHistogram(data):
    #Distribution Plot
    ###TODO: Implement nbins
    fig = px.histogram(data)
    return fig

def createStackedBarChart(data):
    #Mostly from the previous year
    data.apply(pd.value_counts())
    resultData = []
    for i in range(data.shape[0]):
        bar = go.Bar(
            x = np.asArray(data.columns),
            y = data.values[i],
            name = str(data.index[i])
        )
        resultData.append(bar)
    return {
        'data':resultData,
        'layout': go.Layout()
        }

def createScatterPlot(data):
    fig = px.scatter(data)
    ###TODO: Make look nice
    return fig

def createTwoDPlot(data):
    fig = px.line(df)
    ###TODO: Make look nice
    return  fig
