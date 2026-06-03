# Obesidade
### 🩺 Analisador de Perfil de Saúde e Estilo de Vida

Este projeto utiliza Aprendizado de Máquina Não Supervisionado para segmentar perfis de saúde com base em hábitos alimentares, atividade física e dados demográficos. O objetivo é ir além do IMC tradicional, identificando grupos comportamentais (clusters) que permitem intervenções de saúde mais personalizadas.

### 🚀 Funcionalidades

Segmentação em 5 Clusters: Identificação de perfis que variam desde o "Sênior Sedentário" até a "Obesidade Tipo III".

Interface Interativa: Simulador construído em Streamlit que permite ao usuário inserir seus dados e descobrir seu perfil em tempo real.

Visualização Científica: Abas dedicadas para análise de redução de dimensionalidade (PCA, t-SNE, UMAP) e hierarquia (Dendrograma).

Sensibilidade Decimal: Modelo calibrado para entender nuances de hábito (ex: alguém que não é 100% sedentário nem 100% ativo).

### 🛠️ Tecnologias Utilizadas

Linguagem: Python 3.8+

Bibliotecas de ML: Scikit-Learn, Scipy

Visualização: Plotly, Matplotlib

Interface: Streamlit

Processamento: Pandas, Numpy, Joblib

### 📁 Estrutura do Repositório

obesidade.py: Código-fonte da aplicação Streamlit.

modelo_cluster_obesidade_vtl.pkl: Pipeline completo (Preprocessor + K-Means) e coordenadas espaciais pré-calculadas.

requirements.txt: Lista de dependências para rodar o projeto.

Projeto - Obesidade.pdf: Documentação técnica detalhada com hipóteses e conclusões.

### 📊 Os 5 Clusters Identificados

Cluster 0 - Sênior Sedentário: Indivíduos mais velhos, dependentes de automóveis.

Cluster 1 - Fitness / Saudável: Jovem, alta atividade física e dieta equilibrada.

Cluster 2 - Jovem / Dieta Pobre: Estudantes com baixo consumo de vegetais.

Cluster 3 - Meal Skipper: Rotina alimentar desregulada (pula refeições).

Cluster 4 - Alto Risco (Obesidade III): Estado metabólico isolado com alto IMC e influência genética.

### 🔧 Como Executar

Certifique-se de ter o Python instalado.

Instale as dependências:

Bash

pip install -r requirements.txt
Execute a aplicação:

Bash

streamlit run obesidade.py

### 📝 Nota sobre os Dados

Este projeto utiliza o dataset de níveis de obesidade da UCI (México, Peru e Colômbia). Para garantir o equilíbrio das classes e a precisão do modelo, 77% dos dados foram gerados sinteticamente via SMOTE, permitindo que o sistema aprenda as "fronteiras suaves" entre os diferentes estilos de vida.
