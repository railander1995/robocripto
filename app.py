
import streamlit as st
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
import numpy as np

st.set_page_config(page_title="RoboCrypto IA", layout="wide")
st.title("🤖 Robô Cripto com IA - Análise com CoinMarketCap API")

@st.cache_resource
def treinar_modelo():
    X = np.array([
        [1e6, 2e6, 12.3],
        [5e6, 1.5e6, 8.4],
        [2e7, 0.5e6, -5.2],
        [3e6, 3e6, 6.1],
        [8e6, 1.2e6, 7.9],
        [1e7, 1.1e6, -1.0]
    ])
    y = [1, 1, 0, 1, 1, 0]
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X, y)
    return modelo

modelo = treinar_modelo()

def buscar_criptos_cmc():
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
    headers = {
        "Accepts": "application/json",
        "X-CMC_PRO_API_KEY": "5b3fec01-ef41-4918-a9c2-0f991fccf35a"
    }
    params = {
        "start": "1",
        "limit": "50",
        "convert": "USD"
    }
    try:
        r = requests.get(url, headers=headers, params=params, timeout=10)
        r.raise_for_status()
        return r.json().get("data", [])
    except requests.exceptions.RequestException as e:
        st.error(f"Erro na conexão com a CoinMarketCap API: {e}")
        return []

dados = buscar_criptos_cmc()
if not isinstance(dados, list) or not dados:
    st.warning("Nenhum dado disponível no momento. Tente novamente mais tarde.")
    st.stop()

resultados = []

for moeda in dados:
    quote = moeda.get("quote", {}).get("USD", {})
    mc = quote.get("market_cap", 0)
    vol = quote.get("volume_24h", 0)
    var = quote.get("percent_change_24h", 0)

    if all([
        isinstance(mc, (int, float)) and mc > 0,
        isinstance(vol, (int, float)) and vol > 0,
        isinstance(var, (int, float))
    ]):
        entrada = pd.DataFrame([[mc, vol, var]], columns=['market_cap', 'volume_24h', 'price_change_24h'])
        try:
            potencial = modelo.predict_proba(entrada)[0][1] * 100
            resultados.append({
                'Nome': moeda.get('name', ''),
                'Símbolo': moeda.get('symbol', '').upper(),
                'Preço (USD)': quote.get('price', 0),
                'Market Cap': mc,
                'Volume 24h': vol,
                'Variação 24h (%)': var,
                'Potencial (%)': round(potencial, 2)
            })
        except Exception as e:
            st.warning(f"Erro ao calcular potencial para {moeda.get('name', '???')}: {e}")
            continue

df = pd.DataFrame(resultados)
if df.empty or 'Potencial (%)' not in df.columns:
    st.warning("Nenhum dado processado com sucesso. Tente novamente mais tarde.")
    st.stop()

df_filtrado = df[df['Potencial (%)'] > 50].sort_values(by='Potencial (%)', ascending=False)

st.dataframe(df_filtrado, use_container_width=True)
