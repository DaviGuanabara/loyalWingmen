import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import export_text
import graphviz
from sklearn.tree import export_graphviz

# Lendo o arquivo Excel em um DataFrame usando pandas
file_path = 'output_3_hiddens.xlsx'
data_frame = pd.read_excel(file_path)

# Dividindo os dados em recursos (X) e rótulos (y)
X = data_frame[['frequency', 'learning_rate', 'hiddens_1', 'hiddens_2', 'hiddens_3']]
y = data_frame['score']  # Utilizando a coluna "score" como o alvo/target

# Dividindo os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando a árvore de decisão para regressão
regressor = DecisionTreeRegressor()

# Treinando o modelo com os dados de treinamento
regressor.fit(X_train, y_train)

# Realizando predições nos dados de teste
y_pred = regressor.predict(X_test)

# Calculando o erro médio quadrático (Mean Squared Error) do modelo
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Extraindo o impacto de cada coluna na predição
feature_importance = dict(zip(X.columns, regressor.feature_importances_))
print("Impacto de cada coluna na predição:")
for feature, importance in feature_importance.items():
    print(f"{feature}: {importance}")

# Mostrando a árvore de decisão
dot_data = export_graphviz(regressor, out_file=None, 
                           feature_names=X.columns, filled=True, rounded=True, special_characters=True)  
graph = graphviz.Source(dot_data)  
graph.view()
