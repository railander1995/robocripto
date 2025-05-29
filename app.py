
import streamlit as st
import pandas as pd
import pickle
import requests

st.set_page_config(page_title="RoboCrypto IA", layout="wide")
st.title("🤖 Robô Cripto com IA - Detecção de Moedas com Potencial")

@st.cache_data
def carregar_modelo():
    with open("modelo_ia.pkl", "rb") as file:
        return pickle.load(file)

modelo = carregar_modelo()

def buscar_criptos():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_asc",
        "per_page": 50,
        "page": 1,
        "sparkline": False
    }
    r = requests.get(url, params=params)
    return r.json()

dados = buscar_criptos()
resultados = []

for moeda in dados:
    mc = moeda.get('market_cap', 0)
    vol = moeda.get('total_volume', 0)
    var = moeda.get('price_change_percentage_24h', 0)

    if mc and vol and var is not None:
        entrada = pd.DataFrame([[mc, vol, var]], columns=['market_cap', 'volume_24h', 'price_change_24h'])
        potencial = modelo.predict_proba(entrada)[0][1] * 100

        resultados.append({
            'Nome': moeda['name'],
            'Símbolo': moeda['symbol'].upper(),
            'Preço (USD)': moeda['current_price'],
            'Market Cap': mc,
            'Volume 24h': vol,
            'Variação 24h (%)': var,
            'Potencial (%)': round(potencial, 2)
        })

df = pd.DataFrame(resultados)
df_filtrado = df[df['Potencial (%)'] > 50].sort_values(by='Potencial (%)', ascending=False)

st.dataframe(df_filtrado, use_container_width=True)
