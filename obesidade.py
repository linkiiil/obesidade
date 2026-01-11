import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

# Configuração da Página
st.set_page_config(page_title="Health Analytics - Perfil de Obesidade", layout="wide")

@st.cache_resource
def carregar_projeto():
    try:
        # Carrega o dicionário exportado do seu notebook
        return joblib.load('modelo_cluster_obesidade_vtl.pkl')
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo do modelo: {e}")
        return None

def main():
    p = carregar_projeto()
    if not p: return

    st.title("🩺 Analisador de Estilo de Vida e Saúde")
    
    tab_pred, tab_viz = st.tabs(["🔍 Descobrir Meu Perfil", "📊 Painel Técnico de Clusters"])

    with tab_pred:
        # NOTA DE ORIGEM DOS DADOS
        st.info("**Sobre a Inteligência deste Sistema:** Este modelo não usa apenas dados reais. Para ser mais preciso, ele foi 'ensinado' com um dataset expandido artificialmente (via técnica SMOTE), o que permite que ele entenda perfis que estão entre as categorias comuns.")

        # EXPLICAÇÃO SOBRE DECIMAIS
        with st.expander("🤔 Por que posso usar números 'quebrados' (ex: 1.5 ou 2.8)?", expanded=True):
            st.write("""
            Nossa vida não é feita apenas de 'sim' ou 'não'. Se você pratica exercícios às vezes, você não é nem 0 nem 3. 
            Ao usar decimais nos seletores abaixo, você ajuda o modelo a localizar sua **nuance exata de hábito**. 
            Isso torna sua classificação muito mais personalizada e realista do que uma tabela médica rígida.
            """)

        with st.form("formulario_obesidade_final"):
            st.subheader("📋 Insira as suas características")
            c1, c2, c3 = st.columns(3)

            # CORREÇÃO DO ERRO DE SINTAXE AQUI:
            with c1:
                gender = st.selectbox("Gênero", [1.0, 0.0], format_func=lambda x: "Feminino" if x==1 else "Masculino")
                age = st.number_input("Idade", 14.0, 61.0, 25.0, step=0.1)
                height = st.number_input("Altura (m)", 1.45, 2.25, 1.70, step=0.01)
                weight = st.number_input("Peso (kg)", 39.0, 180.0, 75.0, step=0.1)
                fam_hist = st.selectbox("Histórico familiar de sobrepeso?", [1.0, 0.0], format_func=lambda x: "Sim" if x==1 else "Não")

            with c2:
                favc = st.selectbox("Consome alimentos calóricos frequentemente?", [1.0, 0.0], format_func=lambda x: "Sim" if x==1 else "Não")
                fcvc = st.slider("Consumo de vegetais (FCVC) [1 a 3]", 1.0, 3.0, 2.0, step=0.1)
                ncp = st.slider("Refeições principais por dia (NCP)", 1.0, 4.0, 3.0, step=0.1)
                caec_map = {"Às vezes": "Sometimes", "Frequentemente": "Frequently", "Sempre": "Always", "Não": "no"}
                caec = st.selectbox("Consumo de alimentos entre refeições", list(caec_map.keys()))
                smoke = st.selectbox("É fumante?", [1.0, 0.0], format_func=lambda x: "Sim" if x==1 else "Não")

            with c3:
                ch2o = st.slider("Consumo de água (Litros/dia)", 1.0, 3.0, 2.0, step=0.1)
                scc = st.selectbox("Monitoriza o consumo de calorias?", [1.0, 0.0], format_func=lambda x: "Sim" if x==1 else "Não")
                faf = st.slider("Frequência de Atividade física [0 a 3]", 0.0, 3.0, 1.0, step=0.1)
                tue = st.slider("Uso de eletrônicos (Horas/dia)", 0.0, 2.0, 1.0, step=0.1)
                calc_map = {"Não": "no", "Às vezes": "Sometimes", "Frequentemente": "Frequently", "Sempre": "Always"}
                calc = st.selectbox("Consumo de álcool", list(calc_map.keys()))
                mtrans_map = {"Transporte Público": "Public_Transportation", "Automóvel": "Automobile", "Caminhada": "Walking", "Motocicleta": "Motorbike", "Bicicleta": "Bike"}
                mtrans = st.selectbox("Meio de transporte principal", list(mtrans_map.keys()))

            btn = st.form_submit_button("Analisar Meu Perfil")

        if btn:
            bmi = weight / (height ** 2)
            input_df = pd.DataFrame({
                'Gender': [gender], 'Age': [age], 'family_history_with_overweight': [fam_hist],
                'FAVC': [favc], 'FCVC': [fcvc], 'NCP': [ncp], 'CAEC': [caec_map[caec]],
                'SMOKE': [smoke], 'CH2O': [ch2o], 'SCC': [scc], 'FAF': [faf],
                'TUE': [tue], 'CALC': [calc_map[calc]], 'MTRANS': [mtrans_map[mtrans]], 'BMI': [bmi]
            })

            X_transformed = p['preprocessor'].transform(input_df)
            cluster_id = p['kmeans'].predict(X_transformed)[0]

            st.divider()
            nome_cluster, desc_cluster = p['clusters_desc'][cluster_id]
            st.header(f"Seu Perfil Identificado: Cluster {cluster_id}")
            st.subheader(nome_cluster)
            st.write(desc_cluster)
            st.metric("Seu IMC Calculado", f"{bmi:.2f}")
            
    with tab_viz:
        st.header("Análise Espacial e Hierárquica")
        opcao = st.selectbox("Escolha o gráfico:", ["PCA 2D", "PCA 3D", "t-SNE", "UMAP", "Dendrograma"])

        if opcao == "Dendrograma":
            if p.get('linkage_matrix') is not None:
                st.subheader("Dendrograma Truncado (5 Clusters)")
                threshold = p.get('threshold_dendro', 37.0) 
                
                fig, ax = plt.subplots(figsize=(10, 5))
                # Truncate_mode='lastp' com p=5 garante que apenas os 5 clusters finais apareçam
                sch.dendrogram(
                    p['linkage_matrix'],
                    truncate_mode='lastp',
                    p=30, 
                    color_threshold=threshold,
                    ax=ax,
                    show_contracted=True
                )
                ax.axhline(y=threshold, color='r', linestyle='--', label=f'Ponto de Corte: {threshold}')
                ax.set_ylabel("Distância Euclidiana")
                ax.legend()
                st.pyplot(fig)
                st.caption("O gráfico foi truncado para exibir apenas os 5 agrupamentos principais definidos no projeto.")
            else:
                st.warning("Matriz de ligação não disponível.")

        elif opcao in ["PCA 2D", "PCA 3D", "t-SNE", "UMAP"]:
            map_keys = {"PCA 2D": "X_pca2", "PCA 3D": "X_pca3", "t-SNE": "X_tsne", "UMAP": "X_umap"}
            data_viz = p.get(map_keys[opcao])
            if data_viz is not None:
                if opcao == "PCA 3D":
                    fig = px.scatter_3d(x=data_viz[:,0], y=data_viz[:,1], z=data_viz[:,2], color=p['kmeans'].labels_.astype(str))
                else:
                    fig = px.scatter(x=data_viz[:,0], y=data_viz[:,1], color=p['kmeans'].labels_.astype(str))
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()