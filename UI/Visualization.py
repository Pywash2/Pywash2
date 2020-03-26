import dash_html_components as html
import dash_core_components as dcc

def VisualizationUI():
    return html.Div(
        id = 'Visualization',
        children = [
            html.H3(
                children = "Now, let's have a closer look!",
                style = {'textAlign':'center'}
            ),
        ],
        style = {'display': 'none'}
    )
