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
            html.Div(
                id = 'result_data',
                children = [
                ],
#                style = {'width': '50%','textAlign':'center','display': 'inline-block'}
            )
        ],
        style = {'display': 'none'}
    )
