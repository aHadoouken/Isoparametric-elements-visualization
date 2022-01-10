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

# Задание координат узлов 20-узлового элемента в декартовых координатах
x_v = [ -1.1, 1, 1,-1,-0.8, 1, 1.2,-1, 0, 1, 0,-1,-1.3, 1.1, 1,-1, 0, 1, 0,-1]
y_v = [ -1.2,-1, 1, 1,-1.1,-1, 1.2, 1,-1, 0, 1, 0,-0.9,-0.9, 1, 1,-1, 0, 1, 0]
z_v = [ -0.8,-1,-1,-1, 1.2, 1, 1.1, 1,-1,-1,-1,-1, 0.1, -0.1, 0, 0, 1, 1, 1, 1]


# Дополнительные опции для приложения. Выбор режима работы и порта
DEBUG = False
PORT = 8050


#=====================================================================================
# Реализация визуализации. Далее код необходимо оставить неизменным
#=====================================================================================
# Задание координат узлов 20-узлового элемента в нормированных координатах
ksi = np.array([ -1, 1, 1,-1,-1, 1, 1,-1, 0, 1, 0,-1,-1, 1, 1,-1, 0, 1, 0,-1])
eta = np.array([ -1,-1, 1, 1,-1,-1, 1, 1,-1, 0, 1, 0,-1,-1, 1, 1,-1, 0, 1, 0])
zeta = np.array([-1,-1,-1,-1, 1, 1, 1, 1,-1,-1,-1,-1, 0, 0, 0, 0, 1, 1, 1, 1])
x_v = np.array(x_v)
y_v = np.array(y_v)
z_v = np.array(z_v)

# Функция пересчета координат точки из нормированных в декартовые
def form(x, y, z):
    return_int = False
    if type(x) != np.ndarray:
        x = np.array([x])
        y = np.array([y])
        z = np.array([z])
        return_int = True
    N = np.zeros((x.shape[0], x_v.shape[0]))
    N[:, ksi == 0] = 1/4*(1 - x**2).reshape(-1,1)*(1 + np.tensordot(y,eta[ksi == 0],axes=0))*(1 + np.tensordot(z,zeta[ksi == 0],axes=0))
    N[:, eta == 0] = 1/4*(1 - y**2).reshape(-1,1)*(1 + np.tensordot(x,ksi[eta == 0],axes=0))*(1 + np.tensordot(z,zeta[eta == 0],axes=0))
    N[:, zeta == 0] = 1/4*(1 - z**2).reshape(-1,1)*(1 + np.tensordot(y,eta[zeta == 0],axes=0))*(1 + np.tensordot(x,ksi[zeta == 0],axes=0))
    cond = (ksi != 0) & (eta != 0) & (zeta != 0)
    N[:, cond] = 1/8*(1 + np.tensordot(x,ksi[cond],axes=0))*(1 + np.tensordot(y,eta[cond],axes=0))*(1 + np.tensordot(z,zeta[cond],axes=0))*\
    (np.tensordot(x,ksi[cond],axes=0) + np.tensordot(y,eta[cond],axes=0) + np.tensordot(z,zeta[cond],axes=0) - 2)
    x_new = N.dot(x_v)
    y_new = N.dot(y_v)
    z_new = N.dot(z_v)
    if return_int:
        x_new = x_new[0]
        y_new = y_new[0]
        z_new = z_new[0]
    return x_new, y_new, z_new

# Функция визуализации узлов
def make_nodes(row, col, params, norm_form=True):
    if norm_form:
        x = ksi
        y = eta
        z = zeta
    else:
        x = x_v
        y = y_v
        z = z_v
    for i in range(4):
        ind1 = i
        ind2 = (ind1 + 1) % 4
        fig.add_trace(go.Scatter3d(x=[x[ind1], x[ind1 + 8]], y=[y[ind1], y[ind1 + 8]], z=[z[ind1], z[ind1 + 8]], **params), row, col)
        fig.add_trace(go.Scatter3d(x=[x[ind1 + 8], x[ind2]], y=[y[ind1 + 8], y[ind2]], z=[z[ind1 + 8], z[ind2]], **params), row, col)

        fig.add_trace(go.Scatter3d(x=[x[ind1 + 4], x[ind1 + 16]], y=[y[ind1 + 4], y[ind1 + 16]], z=[z[ind1 + 4], z[ind1 + 16]], **params), row, col)
        fig.add_trace(go.Scatter3d(x=[x[ind1 + 16], x[ind2 + 4]], y=[y[ind1 + 16], y[ind2 + 4]], z=[z[ind1 + 16], z[ind2 + 4]], **params), row, col)

        fig.add_trace(go.Scatter3d(x=[x[ind1], x[ind1 + 12]], y=[y[ind1], y[ind1 + 12]], z=[z[ind1], z[ind1 + 12]], **params), row, col)
        fig.add_trace(go.Scatter3d(x=[x[ind1 + 12], x[ind1 + 4]], y=[y[ind1 + 12], y[ind1 + 4]], z=[z[ind1 + 12], z[ind1 + 4]], **params), row, col)

# Функция визуализации ребер
def make_edges(row, col, params):
    point_num = 10
    coord1 = np.linspace(-1, 1, point_num)
    coord2 = np.ones(point_num)
    x, y, z = form(coord1, coord2, coord2)
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, **params), row, col)
    x, y, z = form(coord1, -coord2, coord2)
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, **params), row, col)
    x, y, z = form(coord1, coord2, -coord2)
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, **params), row, col)
    x, y, z = form(coord1, -coord2, -coord2)
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, **params), row, col)

    x, y, z = form(coord2, coord1, coord2)
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, **params), row, col)
    x, y, z = form(-coord2, coord1, coord2)
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, **params), row, col)
    x, y, z = form(coord2, coord1, -coord2)
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, **params), row, col)
    x, y, z = form(-coord2, coord1, -coord2)
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, **params), row, col)

    x, y, z = form(coord2, coord2, coord1)
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, **params), row, col)
    x, y, z = form(-coord2, coord2, coord1)
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, **params), row, col)
    x, y, z = form(coord2, -coord2, coord1)
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, **params), row, col)
    x, y, z = form(-coord2, -coord2, coord1)
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, **params), row, col)

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

x0, y0, z0 = form(0, 0, 0)
fig.add_trace(
    go.Scatter3d(x=[x0], y=[y0], z=[z0],
        mode='markers', 
        marker=dict(
            size=5
        ),
    ),
    row=1, col=2)

params = dict(line=dict(color='darkblue',width=2, dash='dash'), marker=dict(size=2,color=1,colorscale='Viridis'))
make_nodes(1, 1, params, norm_form=True)
params = dict(mode='markers', marker=dict(size=2,color='darkblue'))
make_nodes(1, 2, params, norm_form=False)
params = dict(mode='lines', line=dict(color='darkblue',width=2, dash='dash'))
make_edges(1, 2, params)

fig.update_layout(
    height=700,
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

        x_r, y_r, z_r = form(x, fig.data[0].y[-1], fig.data[0].z[-1])
        fig.data[1].x = list(fig.data[1].x) + [x_r]
        fig.data[1].y = list(fig.data[1].y) + [y_r]
        fig.data[1].z = list(fig.data[1].z) + [z_r]

    elif slider == 'y-slider':
        fig.data[0].x = list(fig.data[0].x) + [fig.data[0].x[-1]]
        fig.data[0].y = list(fig.data[0].y) + [y]
        fig.data[0].z = list(fig.data[0].z) + [fig.data[0].z[-1]]

        x_r, y_r, z_r = form(fig.data[0].x[-1], y, fig.data[0].z[-1])
        fig.data[1].x = list(fig.data[1].x) + [x_r]
        fig.data[1].y = list(fig.data[1].y) + [y_r]
        fig.data[1].z = list(fig.data[1].z) + [z_r]

    elif slider == 'z-slider':
        fig.data[0].x = list(fig.data[0].x) + [fig.data[0].x[-1]]
        fig.data[0].y = list(fig.data[0].y) + [fig.data[0].y[-1]]
        fig.data[0].z = list(fig.data[0].z) + [z]

        x_r, y_r, z_r = form(fig.data[0].x[-1], fig.data[0].y[-1], z)
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
    app.run_server()
