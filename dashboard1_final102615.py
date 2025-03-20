import dash
from dash import dcc, html
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from sklearn import metrics
import joblib  # Para carregar modelos de regressão
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE



# Carregar os dados
df_2019 = pd.read_csv('testData_2019_NorthTower.csv')
df_2019["Date"] = pd.to_datetime(df_2019["Date"])
df_2019.set_index("Date", inplace=True)

# Definir os feriados
feriados_portugal = [
    "2019-01-01", "2019-04-19", "2019-04-21", "2019-04-25", "2019-05-01", 
    "2019-06-10", "2019-08-15", "2019-10-05", "2019-11-01", "2019-12-01", 
    "2019-12-08", "2019-12-25"
]

df_2019["Feriado"] = df_2019.index.strftime("%Y-%m-%d").isin(feriados_portugal).astype(int)
df_2019["Dia_da_Semana_Num"] = df_2019.index.dayofweek.map(lambda x: (x + 1) % 7)
df_2019["Dia"] = df_2019.index.day
df_2019["Fim_de_Semana"] = df_2019["Dia_da_Semana_Num"].isin([0, 6]).astype(int)
df_2019["Hora"] = df_2019.index.hour
df_2019["Ano"] = df_2019.index.year
df_2019["Mês"] = df_2019.index.month

df_2019["Aulas"] = df_2019.apply(lambda row: 1 if (row["Dia_da_Semana_Num"] in [1, 2, 3, 4, 5] and 8 <= row["Hora"] <= 18 and row["Feriado"] == 0 and row["Mês"] in [1, 2, 3, 4, 5, 6, 9, 10, 11, 12]) else 0, axis=1)

df_2019["Hora_sin"] = np.sin(2 * np.pi * df_2019.index.hour / 24)
df_2019["Hora_cos"] = np.cos(2 * np.pi * df_2019.index.hour / 24)

# Somando a potência usada por mês
df_monthly_power = df_2019.groupby('Mês')['North Tower (kWh)'].sum().reset_index()

# Carregar o modelo de regressão
regression_model = joblib.load('modelo_rf.pkl')

# Carregar o modelo de Rede Neural
mlp_regressor = joblib.load('mlp_regressor.pkl')

# Simulação de um dataset
df = pd.DataFrame({
    "temperatura": df_2019["temp_C"],
    "radiacao": df_2019["solarRad_W/m2"],
    "aulas": df_2019["Aulas"],
    "fim_de_semana": df_2019["Fim_de_Semana"] ,
    "consumo": df_2019["North Tower (kWh)"],  # Consumo de energia em kWh (da variável original)
    "umidade": df_2019["HR"],  # Umidade relativa do ar
    "vento": df_2019["windSpeed_m/s"],  # Velocidade do vento
    "presao": df_2019["pres_mbar"],  # Pressão atmosférica
    "chuva": df_2019["rain_mm/h"]  # Precipitação de chuva em mm/h
})

valid_features = ['Hora', 'temp_C', 'solarRad_W/m2', 'Dia_da_Semana_Num', 'Aulas', 'Fim_de_Semana', 'Mês', 'Hora_sin', 'Hora_cos']

# Criar o aplicativo Dash
app = dash.Dash(__name__)

# Layout
app.layout = html.Div(style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f9f9f9', 'padding': '20px'}, children=[
    # Cabeçalho
    html.Div([
        html.Img(src='IST_Logo.png', style={'height': '50px', 'display': 'inline-block', 'marginRight': '10px'}), 
        html.H1('Dashboard - Energy Monitoring 2019', style={'color': '#007acc', 'display': 'inline-block', 'verticalAlign': 'middle'}),
    ], style={'textAlign': 'center', 'paddingBottom': '20px'}),

    # Layout com os dados reais à esquerda e o modelo à direita
    html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'padding': '20px'}, children=[

        # Dados Reais à Esquerda
        html.Div([
            # Título para Consumo de Energia
            html.H1('Consumo de Energia kWh', style={'color': '#007acc', 'display': 'inline-block', 'verticalAlign': 'middle'}),

            # Gráfico dos Dados Reais
            dcc.Tab(label='Consumo de Energia kWh', children=[ 
                html.Div([
                    dcc.Graph(id='raw-data-graph', figure={}),
                ])
            ]),

            # Título para Consumo de Energia
            html.H1('Todas as Features', style={'color': '#007acc', 'display': 'inline-block', 'verticalAlign': 'middle'}),

            # Dropdown para escolher a variável e tipo de gráfico
            dcc.Dropdown(
                id='graph-type-dropdown',
                options=[
                    {'label': 'Gráfico de Linha', 'value': 'line'},
                    {'label': 'Gráfico de Barras', 'value': 'bar'},
                    {'label': 'Gráfico de Dispersão', 'value': 'scatter'},
                    {'label': 'Histograma', 'value': 'histogram'},
                    {'label': 'Boxplot', 'value': 'box'}
                ],
                value='line',  # Valor inicial
                style={'width': '50%', 'padding': '10px'}
            ), 
            dcc.Dropdown(
                id='variable-dropdown',
                options=[
                    {'label': 'Temperatura (°C)', 'value': 'temp_C'},
                    {'label': 'Radiação Solar (W/m²)', 'value': 'solarRad_W/m2'},
                    {'label': 'Aulas', 'value': 'Aulas'},
                    {'label': 'Fim de Semana', 'value': 'Fim_de_Semana'},
                    
                ],
                value='temp_C',  # Valor inicial
                style={'width': '50%'}
            ),
            # Gráficos e Tabelas
            dcc.Graph(id='graph-output'),

            html.Label("Escolha as Features:"),
            dcc.Dropdown(
                id="feature-selector",
                options=[
                    {"label": "Temperatura (°C)", "value": "temperatura"},
                    {"label": "Radiação Solar (W/m²)", "value": "radiacao"},
                    {"label": "Aulas", "value": "aulas"},
                    {"label": "Fim de Semana", "value": "fim_de_semana"},
                    {'label': 'Vento', 'value': 'vento'},
                    {'label': 'Pressão', 'value': 'presao'},
                    {'label': 'Chuva', 'value': 'chuva'},
                    {'label': 'Mês', 'value': 'Mês'}
                ],
                multi=True,
                value=["temperatura"]  # Default
            ),

            html.Label("Método de Seleção de Features:"),
            dcc.Dropdown(
                id="feature-selection-method",
                options=[
                    {"label": "Nenhum (Todas as Selecionadas)", "value": "none"},
                    {"label": "PCA (Análise de Componentes Principais)", "value": "pca"},
                    {"label": "Seleção por Importância (Random Forest)", "value": "importance"},
                    {"label": "RFE (Recursive Feature Elimination)", "value": "rfe"}
                ],
                value="none"  # Default
            ),

            html.Div(id="output-selection")

            ], style={'width': '48%'}),  # Largura do lado esquerdo

        # Informações sobre o Modelo à Direita
        html.Div([
            # Título para Previsão com Modelo
            html.H1('Previsão com o Modelo Escolhido', style={'color': '#007acc', 'display': 'inline-block', 'verticalAlign': 'middle'}),

            # Dropdown para escolher o modelo de previsão
            dcc.Dropdown(
                id='model-dropdown',
                options=[
                    {'label': 'Random Forest', 'value': 'rf'},
                    {'label': 'Rede Neural', 'value': 'nn'}
                ],
                value='rf',  # Valor inicial
                style={'width': '100%', 'padding': '10px'}
            ),

            # Checkboxes para escolher as métricas
            html.Div([
                dcc.Checklist(
                    id='metrics-dropdown',
                    options=[
                        {'label': 'Erro Médio Absoluto (MAE)', 'value': 'mae'},
                        {'label': 'Erro Médio (MBE)', 'value': 'mbe'},
                        {'label': 'Erro Quadrático Médio (RMSE)', 'value': 'rmse'},
                        {'label': 'R²', 'value': 'r2'}
                    ],
                    value=['mae', 'rmse'],  # Valor inicial
                    labelStyle={'display': 'inline-block', 'marginRight': '10px'}
                ),
            ], style={'padding': '20px', 'textAlign': 'center'}),

            # Gráficos e Tabelas para o Modelo de Regressão
            dcc.Tabs([
                dcc.Tab(label='Previsão com o Modelo', children=[
                    html.Div([
                        dcc.Graph(id='regression-plot', figure={}),
                        html.Div(id='regression-table')
                    ])
                ]),
            ]),

            # Título para Consumo de Energia
            html.H1('Potênica usada ao ano', style={'color': '#007acc', 'display': 'inline-block', 'verticalAlign': 'middle'}),
            html.Div([
                html.H3('Escolha os Meses a Mostrar', style={'color': '#007acc'}),
                dcc.Checklist(
                    id='month-checklist',
                    options=[
                        {'label': 'Todos os Meses', 'value': 'all'},
                        {'label': 'Janeiro', 'value': '1'},
                        {'label': 'Fevereiro', 'value': '2'},
                        {'label': 'Março', 'value': '3'},
                        {'label': 'Abril', 'value': '4'},
                    ],
                    value=['all'],  # Valor inicial selecionado (todos os meses)
                    labelStyle={'display': 'inline-block', 'marginRight': '10px'}
                ),
            ], style={'padding': '20px', 'textAlign': 'center'}),

            # Gráfico de histograma com base na seleção da checklist
            html.Div([  
                dcc.Graph(
                    id='monthly-power-histogram',
                    figure={}
                ),
            ], style={'paddingBottom': '30px'}),

        ], style={'width': '48%'}),  # Largura do lado direito
    ])
])

# Função de callback para gerar o gráfico com base na seleção
@app.callback(
    Output('graph-output', 'figure'),
    [Input('variable-dropdown', 'value'),
     Input('graph-type-dropdown', 'value')]
)
def update_graph(selected_variable, selected_graph_type):
    # Selecionar os dados com base na variável escolhida
    x_data = df_2019.index
    y_data = df_2019[selected_variable]

    # Gerar o gráfico baseado no tipo selecionado
    if selected_graph_type == 'bar':
        figure = {
            'data': [
                go.Bar(x=x_data, y=y_data, name=selected_variable)
            ],
            'layout': {
                'title': f'Gráfico de Barras - {selected_variable}',
                'xaxis': {'title': 'Data'},
                'yaxis': {'title': selected_variable},
            }
        }
    elif selected_graph_type == 'scatter':
        figure = {
            'data': [
                go.Scatter(x=x_data, y=y_data, mode='markers', name=selected_variable)
            ],
            'layout': {
                'title': f'Gráfico de Dispersão - {selected_variable}',
                'xaxis': {'title': 'Data'},
                'yaxis': {'title': selected_variable},
            }
        }
    elif selected_graph_type == 'histogram':
        figure = {
            'data': [
                go.Histogram(x=y_data, name=selected_variable)
            ],
            'layout': {
                'title': f'Histograma - {selected_variable}',
                'xaxis': {'title': selected_variable},
                'yaxis': {'title': 'Frequência'},
            }
        }
    elif selected_graph_type == 'box':
        figure = {
            'data': [
                go.Box(y=y_data, name=selected_variable)
            ],
            'layout': {
                'title': f'Boxplot - {selected_variable}',
                'yaxis': {'title': selected_variable},
            }
        }
    else:
        figure = {
            'data': [
                go.Scatter(x=x_data, y=y_data, mode='lines', name=selected_variable)
            ],
            'layout': {
                'title': f'Gráfico de Linha - {selected_variable}',
                'xaxis': {'title': 'Data'},
                'yaxis': {'title': selected_variable},
            }
        }

    return figure

# Callback para "Dados Brutos"
@app.callback(
    Output('raw-data-graph', 'figure'),
    Input('variable-dropdown', 'value')
)
def update_raw_data_graph(selected_variable):
    # Gráfico de dados brutos
    figure = {
        'data': [
            {'x': df_2019.index, 'y': df_2019['North Tower (kWh)'], 'type': 'line', 'name': selected_variable},
        ],
        'layout': {
            'title': f'Dados Brutos - {selected_variable}',
            'xaxis': {'title': 'Data'},
            'yaxis': {'title': selected_variable},
        }
    }

    return figure

# Callback para "Previsão com o Modelo"
@app.callback(
    Output('regression-plot', 'figure'),
    Output('regression-table', 'children'),
    Input('model-dropdown', 'value'),
    Input('metrics-dropdown', 'value')
)
def update_regression_plot(selected_model, selected_metrics):
    # Previsão com o modelo escolhido
    X = df_2019[valid_features]  # Usando as features válidas
    y = df_2019['North Tower (kWh)']  # Consumo de energia
    
    if selected_model == 'rf':
        predictions = regression_model.predict(X)
    elif selected_model == 'nn':
        predictions = mlp_regressor.predict(X)
    
    # Gráfico de previsão
    figure = {
        'data': [
            {'x': df_2019.index, 'y': y, 'type': 'line', 'name': 'Real', 'line': {'color': '#007acc'}},
            {'x': df_2019.index, 'y': predictions, 'type': 'line', 'name': 'Previsto', 'line': {'color': '#ff6347'}},
        ],
        'layout': {
            'title': 'Previsões vs Real - Consumo de Energia',
            'xaxis': {'title': 'Data'},
            'yaxis': {'title': 'Consumo (kWh)'},
        }
    }
    
    # Cálculo das métricas escolhidas
    table = []
    if 'mae' in selected_metrics:
        mae = metrics.mean_absolute_error(y, predictions)
        table.append(html.Tr([html.Td("Erro Médio Absoluto (MAE)"), html.Td(mae)]))
    if 'mbe' in selected_metrics:
        mbe = np.mean(y - predictions)
        table.append(html.Tr([html.Td("Erro Médio (MBE)"), html.Td(mbe)]))
    if 'rmse' in selected_metrics:
        rmse = np.sqrt(metrics.mean_squared_error(y, predictions))
        table.append(html.Tr([html.Td("Erro Quadrático Médio (RMSE)"), html.Td(rmse)]))
    if 'r2' in selected_metrics:
        r2 = metrics.r2_score(y, predictions)
        table.append(html.Tr([html.Td("R²"), html.Td(r2)]))

    # Criar tabela de métricas
    table = html.Table([html.Tr([html.Th("Métrica"), html.Th("Valor")])] + table)
    
    return figure, table

@app.callback(
    Output('monthly-power-histogram', 'figure'),
    Input('month-checklist', 'value')
)
def update_histogram(selected_months):
    if not selected_months:
        # Nenhum mês selecionado -> Retorna gráfico vazio
        return {
            'data': [],
            'layout': {
                'title': 'Nenhum mês selecionado',
                'xaxis': {'title': 'Mês'},
                'yaxis': {'title': 'Potência Usada (kWh)'},
                'plot_bgcolor': '#f9f9f9',
                'paper_bgcolor': '#f9f9f9',
            }
        }

    # Criar lista de dados para o gráfico
    data_list = []
    show_all = 'all' in selected_months

    # Se "All" for selecionado, adicionamos a barra do ano inteiro
    if show_all:
        total_power = df_2019['North Tower (kWh)'].sum()
        data_list.append({
            'x': ['Ano'],
            'y': [total_power],
            'type': 'bar',
            'name': 'Potência Usada no Ano',
            'marker': {'color': '#ff7f0e'}
        })

    # Filtrando apenas os meses específicos
    selected_months = [int(m) for m in selected_months if m.isdigit()]

    if selected_months:
        # Pegando os dados dos meses selecionados
        data = df_2019[df_2019['Mês'].isin(selected_months)]
        monthly_data = data.groupby('Mês')['North Tower (kWh)'].sum()

        # Adicionando cada mês como uma barra separada
        for month, power in monthly_data.items():
            data_list.append({
                'x': [month],
                'y': [power],
                'type': 'bar',
                'name': f'Mês {month}',
                'marker': {'color': '#1f77b4'}
            })

    # Criando o layout do gráfico
    figure = {
        'data': data_list,
        'layout': {
            'title': 'Potência Usada',
            'xaxis': {
                'title': 'Período',
                'tickvals': list(range(1, 4)) + (['Ano'] if show_all else []),
                'ticktext': ['Jan', 'Fev', 'Mar', 'Abr'] + (['Ano'] if show_all else [])
            },
            'yaxis': {'title': 'Potência Usada (kWh)'},
            'plot_bgcolor': '#f9f9f9',
            'paper_bgcolor': '#f9f9f9',
        }
    }

    return figure

@app.callback(
    Output("output-selection", "children"),
    [Input("feature-selector", "value"), Input("feature-selection-method", "value")]
)
def update_feature_selection(selected_features, method):
    return f"Features Selecionadas: {selected_features}, Método: {method}"

# Rodar o servidor
if __name__ == '__main__':
    app.run_server()