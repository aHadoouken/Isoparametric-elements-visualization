# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
app = dash.Dash(__name__)
#=====================================================================================
# Задание входных данных и опций приложения
#=====================================================================================

# Задание координат узлов 8-узлового элемента в декартовых координатах
x_v = [-1.2, 1,   1,-0.7,-1, 1, 1,-1]
y_v = [-1.5,-0.9, 1, 1.2,-1,-1, 1, 1]
z_v = [-0.8,-1,  -1,-1.3, 1, 1, 1.3, 1]


# Дополнительные опции для приложения. Выбор режима работы и порта
DEBUG = False
PORT = 8050


#=====================================================================================
# Реализация визуализации. Далее код необходимо оставить неизменным
#=====================================================================================
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'surface'}, {'type': 'surface'}]])

fig.add_trace(
    go.Scatter3d(x=[0.0], y=[0.0], z=[0.0],
        mode='markers', 
        marker=dict(
            size=5
        ),
    ),
    row=1, col=1)

ksi = np.array([ -1, 1, 1,-1,-1, 1, 1,-1])
eta = np.array([ -1,-1, 1, 1,-1,-1, 1, 1])
zeta = np.array([-1,-1,-1,-1, 1, 1, 1, 1])
N = 1/8*(1 + 0*ksi)*(1 + 0*eta)*(1 + 0*zeta)
x0 = np.sum(x_v*N)
y0 = np.sum(y_v*N)
z0 = np.sum(z_v*N)
fig.add_trace(
    go.Scatter3d(x=[x0], y=[y0], z=[z0],
        mode='markers', 
        marker=dict(
            size=5
        ),
    ),
    row=1, col=2)

params = dict(line=dict(color='darkblue',width=2, dash='dash'), marker=dict(size=2,color=1,colorscale='Viridis'))
for i in range(4):
    ind1 = i
    ind2 = (ind1 + 1) % 4
    fig.add_trace(go.Scatter3d(x=[x_v[ind1], x_v[ind2]], y=[y_v[ind1], y_v[ind2]], z=[z_v[ind1], z_v[ind2]], **params), 1, 2)
    fig.add_trace(go.Scatter3d(x=[x_v[ind1 + 4], x_v[ind2 + 4]], y=[y_v[ind1 + 4], y_v[ind2 + 4]], z=[z_v[ind1 + 4], z_v[ind2 + 4]], **params), 1, 2)
    fig.add_trace(go.Scatter3d(x=[x_v[ind1], x_v[ind1 + 4]], y=[y_v[ind1], y_v[ind1 + 4]], z=[z_v[ind1], z_v[ind1 + 4]], **params), 1, 2)

for i in range(4):
    ind1 = i
    ind2 = (ind1 + 1) % 4
    fig.add_trace(go.Scatter3d(x=[ksi[ind1], ksi[ind2]], y=[eta[ind1], eta[ind2]], z=[zeta[ind1], zeta[ind2]], **params), 1, 1)
    fig.add_trace(go.Scatter3d(x=[ksi[ind1 + 4], ksi[ind2 + 4]], y=[eta[ind1 + 4], eta[ind2 + 4]], z=[zeta[ind1 + 4], zeta[ind2 + 4]], **params), 1, 1)
    fig.add_trace(go.Scatter3d(x=[ksi[ind1], ksi[ind1 + 4]], y=[eta[ind1], eta[ind1 + 4]], z=[zeta[ind1], zeta[ind1 + 4]], **params), 1, 1)

fig.update_layout(
    #width=1000,
    height=700,
    #paper_bgcolor="LightSteelBlue",
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0,
    ),
    showlegend=False
)

fig_start = go.Figure(fig)

app.layout = html.Div([
    html.Button('Reset graphs', id='btn1'),
    dcc.Graph(
        id='graph',
    ),
    html.Div(
        children='X axis',
        style={
            'textAlign': 'center',
        }
    ),
    dcc.Slider(
        id='x-slider',
        min=-1,
        max=1,
        value=0,
        step=0.05,
        tooltip={"placement": "bottom", "always_visible": True},
        updatemode='drag'
    ),
    html.Div(
        children='Y axis',
        style={
            'textAlign': 'center',
        }
    ),
    dcc.Slider(
        id='y-slider',
        min=-1,
        max=1,
        value=0,
        step=0.05,
        tooltip={"placement": "bottom", "always_visible": True},
        updatemode='drag'
    ),
    html.Div(
        children='Z axis',
        style={
            'textAlign': 'center',
        }
    ),
    dcc.Slider(
        id='z-slider',
        min=-1,
        max=1,
        value=0,
        step=0.05,
        tooltip={"placement": "bottom", "always_visible": True},
        updatemode='drag'
    )
])

@app.callback(
    Output('graph', 'figure'),
    Input('x-slider', 'value'),
    Input('y-slider', 'value'),
    Input('z-slider', 'value'),
    Input('btn1', 'n_clicks'))
def update_figure(x, y, z, btn1):
    ctx = dash.callback_context
    global fig
    global fig_start
    if not ctx.triggered:
        return fig
    slider = ctx.triggered[0]['prop_id'].split('.')[0]

    if slider == 'btn1':
        fig = go.Figure(fig_start)
        return fig
    elif slider == 'x-slider':
        fig.data[0].x = list(fig.data[0].x) + [x]
        fig.data[0].y = list(fig.data[0].y) + [fig.data[0].y[-1]]
        fig.data[0].z = list(fig.data[0].z) + [fig.data[0].z[-1]]

        N = 1/8*(1 + x*ksi)*(1 + fig.data[0].y[-1]*eta)*(1 + fig.data[0].z[-1]*zeta)
        x_r = np.sum(x_v*N)
        y_r = np.sum(y_v*N)
        z_r = np.sum(z_v*N)
        fig.data[1].x = list(fig.data[1].x) + [x_r]
        fig.data[1].y = list(fig.data[1].y) + [y_r]
        fig.data[1].z = list(fig.data[1].z) + [z_r]

    elif slider == 'y-slider':
        fig.data[0].x = list(fig.data[0].x) + [fig.data[0].x[-1]]
        fig.data[0].y = list(fig.data[0].y) + [y]
        fig.data[0].z = list(fig.data[0].z) + [fig.data[0].z[-1]]

        N = 1/8*(1 + fig.data[0].x[-1]*ksi)*(1 + y*eta)*(1 + fig.data[0].z[-1]*zeta)
        x_r = np.sum(x_v*N)
        y_r = np.sum(y_v*N)
        z_r = np.sum(z_v*N)
        fig.data[1].x = list(fig.data[1].x) + [x_r]
        fig.data[1].y = list(fig.data[1].y) + [y_r]
        fig.data[1].z = list(fig.data[1].z) + [z_r]

    elif slider == 'z-slider':
        fig.data[0].x = list(fig.data[0].x) + [fig.data[0].x[-1]]
        fig.data[0].y = list(fig.data[0].y) + [fig.data[0].y[-1]]
        fig.data[0].z = list(fig.data[0].z) + [z]

        N = 1/8*(1 + fig.data[0].x[-1]*ksi)*(1 + fig.data[0].y[-1]*eta)*(1 + z*zeta)
        x_r = np.sum(x_v*N)
        y_r = np.sum(y_v*N)
        z_r = np.sum(z_v*N)
        fig.data[1].x = list(fig.data[1].x) + [x_r]
        fig.data[1].y = list(fig.data[1].y) + [y_r]
        fig.data[1].z = list(fig.data[1].z) + [z_r]
    return fig

@app.callback(
    Output('x-slider', 'value'),
    Output('y-slider', 'value'),
    Output('z-slider', 'value'),
    Input('btn1', 'n_clicks'))
def reset_sliders(btn1):
    return 0, 0, 0

if __name__ == '__main__':
    app.run_server(debug=DEBUG, port=PORT)