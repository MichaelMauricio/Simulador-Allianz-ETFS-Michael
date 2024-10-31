import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from fpdf import FPDF
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import io

# Configuración inicial de Streamlit
st.set_page_config(page_title="Simulador de Inversiones - Allianz Patrimonial", layout="wide")

# Función para generar el PDF
def generar_pdf(datos_personales, etfs_seleccionados, pesos, rendimiento, riesgo, simulacion_ahorro, grafico_simulacion):
    pdf = FPDF()
    pdf.add_page()

    # Encabezado
    pdf.set_font("Arial", "B", 16)
    pdf.set_text_color(0, 51, 102)  # Azul oscuro
    pdf.image("Allianz logo.png", x=10, y=8, w=30)  # Logo Allianz en la parte superior izquierda
    pdf.cell(200, 10, "Cotización de Inversión Patrimonial", ln=True, align="C")
    pdf.ln(20)

    # Sección 1: Datos Personales
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, "Datos Personales", ln=True, align="L")
    pdf.set_font("Arial", "", 12)
    for campo, valor in datos_personales.items():
        pdf.cell(200, 10, f"{campo}: {valor}", ln=True)
    pdf.ln(10)  # Espacio adicional entre secciones

    # Sección 2: Simulación de Cartera
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, "Simulación de Cartera", ln=True, align="L")
    pdf.set_font("Arial", "", 12)
    for etf, peso in zip(etfs_seleccionados, pesos):
        pdf.cell(200, 10, f"{etf}: {peso}%", ln=True)
    pdf.ln(10)  # Espacio adicional

    # Rendimiento y Riesgo de la Cartera
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, "Rendimiento Esperado de la Cartera:", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(200, 10, f"{rendimiento}%", ln=True)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, "Riesgo de la Cartera:", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(200, 10, f"{riesgo}%", ln=True)
    pdf.ln(15)  # Espacio adicional para separar secciones

    # Sección 3: Simulador de Ahorro e Inversión Personalizada
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, "Simulador de Ahorro e Inversión Personalizada", ln=True, align="L")
    pdf.set_font("Arial", "", 12)
    for campo, valor in simulacion_ahorro.items():
        pdf.cell(200, 10, f"{campo}: {valor}", ln=True)
    pdf.ln(10)  # Espacio adicional

    # Análisis de Rendimiento
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, "Análisis de Rendimiento", ln=True, align="L")
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, f"El rendimiento promedio anual de la cartera seleccionada es de {rendimiento}%, con un riesgo asociado de {riesgo}%.")
    pdf.ln(10)

    # Gráfica del Simulador de Inversión al final de la página
    pdf.add_page()  # Nueva página para el gráfico del simulador de inversión
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, "Crecimiento de la Inversión", ln=True, align="L")
    pdf.image(grafico_simulacion, x=10, y=30, w=180)  # Insertar gráfica de comparación de inversión vs. ahorro
    pdf.ln(95)  # Añadir espacio adicional después de la imagen

    # Guardar el PDF en un objeto de BytesIO para su descarga
    pdf_output = io.BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)
    return pdf_output

# Función auxiliar para guardar gráficas
def guardar_grafico(fig):
    img = io.BytesIO()
    fig.savefig(img, format="PNG")
    img.seek(0)
    return img

# Nombres completos de los ETFs
etf_names = {
    "SPY": "SPDR S&P 500 ETF Trust",
    "QQQ": "Invesco QQQ Trust",
    "EEM": "iShares MSCI Emerging Markets ETF",
    "IVV": "iShares Core S&P 500 ETF",
    "IEMG": "iShares Core MSCI Emerging Markets ETF",
    "VOO": "Vanguard S&P 500 ETF",
    "VTI": "Vanguard Total Stock Market ETF",
    "BND": "Vanguard Total Bond Market ETF",
    "GLD": "SPDR Gold Shares"
}

# Crear lista de ETFs con nombres para los menús de selección
etfs_with_names = [f"{ticker} ({name})" for ticker, name in etf_names.items()]
etfs = list(etf_names.keys())  # Lista de tickers para el código interno

# Página de Bienvenida
with st.container():
    st.title("¿Estás buscando planes de inversión?")
    st.subheader("Maximiza el crecimiento y seguridad de tu patrimonio")
    st.write("Agenda tu cita con un asesor y recibe un plan a tu medida.")

    # Crear formulario utilizando componentes de Streamlit
    nombre = st.text_input("Nombre Completo")
    telefono = st.text_input("Número de Teléfono")
    email = st.text_input("Correo Electrónico")
    ciudad = st.text_input("Ciudad / Estado")
    edad = st.number_input("Edad", min_value=18, max_value=100, step=1)

    # Aceptación de política de privacidad
    acepta_privacidad = st.checkbox("Acepto el Aviso de Privacidad")
    if acepta_privacidad:
        st.write("Al compartir tus datos, confirmas que tienes capacidad legal para contratar y que eres mayor de edad.")

    # Botón para enviar los datos
    if st.button("ENVIAR"):
        if nombre and telefono and email and ciudad and edad and acepta_privacidad:
            st.session_state["user_data"] = {
                "Nombre": nombre,
                "Teléfono": telefono,
                "Email": email,
                "Ciudad": ciudad,
                "Edad": edad
            }
            st.success(f"Gracias {nombre}, hemos recibido tu información.")
        else:
            st.error("Por favor, completa todos los campos antes de enviar el formulario.")

# Continuar a la aplicación solo si los datos están registrados
if "user_data" in st.session_state:
    st.write(f"**Bienvenido, {st.session_state['user_data']['Nombre']}!**")

    # Sidebar para seleccionar los ETFs y el periodo de tiempo
    st.sidebar.header("Selecciona las Opciones")
    selected_etfs = st.sidebar.multiselect("Selecciona uno o más ETFs", etfs_with_names, [f"SPY ({etf_names['SPY']})", f"QQQ ({etf_names['QQQ']})"])

    # Establecer la fecha de inicio para cubrir hasta 10 años
    start_date = st.sidebar.date_input("Fecha de inicio", datetime.now() - timedelta(days=365 * 12))
    end_date = st.sidebar.date_input("Fecha de fin", datetime.now())

    # Obtener solo los tickers seleccionados (sin los nombres completos)
    selected_etfs = [ticker.split()[0] for ticker in selected_etfs]

    # Función para descargar datos
    @st.cache_data
    def download_data(ticker, start, end):
        try:
            data = yf.download(ticker, start=start, end=end)
            if data.empty:
                st.warning(f"No se encontraron datos para el ticker {ticker}.")
            return data
        except Exception as e:
            st.error(f"Error al descargar datos de {ticker}: {e}")
            st.stop()
            return pd.DataFrame()

    # Descarga de datos para cada ETF seleccionado y gráfico de desempeño comparativo
    if selected_etfs:
        st.header(f"Desempeño Comparativo de {', '.join(selected_etfs)}")

        precios = pd.DataFrame()
        for ticker in selected_etfs:
            data = download_data(ticker, start_date, end_date)
            precios[ticker] = data['Close']

        # Normalizar los precios al valor inicial (1000 USD) para comparar desempeños
        precios_normalizados = precios / precios.iloc[0] * 1000

        # Mostrar gráfico comparativo con `st.line_chart`
        st.line_chart(precios_normalizados, use_container_width=True)

    # Cálculos de rendimiento y riesgo
    st.header("Rendimiento y Riesgo")
    periodos = {
        "1 mes": 21, "3 meses": 63, "6 meses": 126, "1 año": 252,
        "YTD": (datetime.now() - datetime(datetime.now().year, 1, 1)).days,
        "3 años": 252*3, "5 años": 252*5, "10 años": 252*10
    }

    # Inicializar diccionarios para almacenar los resultados
    rendimiento = {periodo: {} for periodo in periodos}
    volatilidad = {periodo: {} for periodo in periodos}

    # Calcular rendimiento y volatilidad para cada ETF y periodo
    for ticker in selected_etfs:
        for periodo, days in periodos.items():
            if len(precios[ticker].dropna()) >= days:
                data_periodo = precios[ticker].dropna().tail(days)
                rendimiento[periodo][ticker] = round((data_periodo.iloc[-1] / data_periodo.iloc[0] - 1) * 100, 2)
                volatilidad[periodo][ticker] = round(data_periodo.pct_change().std() * np.sqrt(252) * 100, 2)
            else:
                # Mostrar advertencia si no hay suficientes datos para el periodo
                rendimiento[periodo][ticker] = "No hay datos suficientes"
                volatilidad[periodo][ticker] = "No hay datos suficientes"

    # Crear DataFrames para mostrar los resultados de rendimiento y volatilidad
    rend_df = pd.DataFrame(rendimiento)
    vol_df = pd.DataFrame(volatilidad)

    # Mostrar tablas de rendimiento y volatilidad
    st.subheader("Rendimiento (%)")
    st.table(rend_df)

    st.subheader("Volatilidad (%)")
    st.table(vol_df)

    # Simulación de cartera
    st.header("Simulación de Cartera")
    num_assets = st.slider("Número de ETFs en la cartera", 1, len(etfs), 3)
    selected_etfs_for_portfolio = st.multiselect("Selecciona los ETFs para la cartera", etfs_with_names, etfs_with_names[:num_assets])

    # Obtener solo los tickers seleccionados (sin los nombres completos) para la simulación de cartera
    selected_etfs_for_portfolio = [ticker.split()[0] for ticker in selected_etfs_for_portfolio]

    # Configuración de la tabla de porcentajes
    if selected_etfs_for_portfolio:
        initial_percentage = round(100 / len(selected_etfs_for_portfolio), 2)
        percentages = {etf: initial_percentage for etf in selected_etfs_for_portfolio}

        # Actualizar los datos en session_state solo si los ETFs seleccionados cambian
        if "selected_etfs_for_portfolio" not in st.session_state or st.session_state.selected_etfs_for_portfolio != selected_etfs_for_portfolio:
            st.session_state.selected_etfs_for_portfolio = selected_etfs_for_portfolio
            percentages_df = pd.DataFrame(list(percentages.items()), columns=["ETF", "Porcentaje (%)"])
            percentages_df["Porcentaje (%)"] = percentages_df["Porcentaje (%)"].astype(float)
            percentages_df.index += 1
            st.session_state.percentages_df = percentages_df
        else:
            percentages_df = st.session_state.percentages_df

        # Editor de datos interactivo (solo una vez)
        edited_percentages_df = st.data_editor(percentages_df, use_container_width=True, key="editor")

        # Comprobación de la suma de porcentajes
        total_percentage = edited_percentages_df["Porcentaje (%)"].sum()
        
        # Mostrar un mensaje y un botón de ajuste si la suma no es 100%
        if total_percentage != 100:
            st.error("La suma debe ser igual al 100%.")
            if st.button("Ajustar porcentajes automáticamente"):
                # Calcular los porcentajes ajustados proporcionalmente
                scale_factor = 100 / total_percentage
                adjusted_percentages = edited_percentages_df["Porcentaje (%)"] * scale_factor
                adjusted_percentages = adjusted_percentages.round(2)
                
                # Ajustar para que sume 100% después del redondeo
                difference = 100.00 - adjusted_percentages.sum()
                adjusted_percentages.iloc[-1] += difference  # Ajusta el último valor para asegurar la suma

                # Actualizar los valores en la tabla principal y guardarlos en session_state
                st.session_state.percentages_df["Porcentaje (%)"] = adjusted_percentages
                st.success("Los porcentajes han sido ajustados y guardados.")
        else:
            st.success("La suma de los porcentajes es igual a 100%.")

        weights = np.array(st.session_state.percentages_df["Porcentaje (%)"]) / 100

        # Descargar datos de los ETFs seleccionados
        cartera_data = pd.DataFrame()
        for ticker in selected_etfs_for_portfolio:
            cartera_data[ticker] = download_data(ticker, start_date, end_date)['Close']

        # Verificar que los pesos coincidan con el número de ETFs seleccionados
        if len(weights) != len(cartera_data.columns):
            st.error("El número de pesos no coincide con el número de ETFs seleccionados. Ajusta la selección y los pesos.")
        else:
            # Calcular rendimientos diarios y matriz de covarianza
            daily_returns = cartera_data.pct_change().dropna()
            cov_matrix = daily_returns.cov() * 252
            expected_return = round(daily_returns.mean().dot(weights) * 252 * 100, 2)  # Rendimiento esperado en porcentaje
            portfolio_volatility = round(np.sqrt(weights.T.dot(cov_matrix).dot(weights)) * 100, 2)

            st.write("*Rendimiento Esperado de la Cartera (%):*", expected_return)
            st.write("*Riesgo de la Cartera (Volatilidad %):*", portfolio_volatility)

            # Simulación de Ahorro e Inversión Personalizada con ajuste al crecimiento final
            st.header("Simulador de Ahorro e Inversión Personalizada")

            # Entradas del simulador
            aportacion_inicial = st.number_input("Aportación inicial", min_value=0, value=1000, step=100)
            aportacion_periodica = st.number_input("Aportación periódica", min_value=0, value=100, step=10)
            frecuencia_aportacion = st.selectbox("Frecuencia de aportaciones", ["Mensual", "Semestral", "Anual"])
            horizonte_inversion = st.selectbox("Horizonte de inversión (años)", [5, 10, 15, 20])

            # Convertir frecuencia a número de aportaciones por año
            frecuencias = {"Mensual": 12, "Semestral": 2, "Anual": 1}
            num_aportaciones_anuales = frecuencias[frecuencia_aportacion]

            # Ajustar la tasa de rendimiento a la frecuencia de aportaciones
            rendimiento_anual = expected_return / 100
            tasa_aportacion = (1 + rendimiento_anual) ** (1 / num_aportaciones_anuales) - 1  # Tasa para cada aportación

            # Simulación de crecimiento de inversión con rendimiento y sin rendimiento
            patrimonio_inversion = [aportacion_inicial]
            patrimonio_ahorro = [aportacion_inicial]

            for _ in range(horizonte_inversion * num_aportaciones_anuales):
                # Inversión con rendimiento
                nuevo_valor_inversion = patrimonio_inversion[-1] + aportacion_periodica
                nuevo_valor_inversion *= (1 + tasa_aportacion)
                patrimonio_inversion.append(nuevo_valor_inversion)
                
                # Ahorro sin rendimiento
                patrimonio_ahorro.append(patrimonio_ahorro[-1] + aportacion_periodica)

            # Crear DataFrame para el gráfico de comparación de inversión vs. ahorro
            inversion_df = pd.DataFrame({
                "Inversión con Rendimiento": patrimonio_inversion,
                "Ahorro sin Rendimiento": patrimonio_ahorro
            }, index=[i / num_aportaciones_anuales for i in range(len(patrimonio_inversion))])

            # Mostrar el gráfico de inversión vs. ahorro con `st.line_chart`
            fig_simulador, ax = plt.subplots()
            ax.plot(inversion_df.index, inversion_df["Inversión con Rendimiento"], label="Inversión con rendimiento")
            ax.plot(inversion_df.index, inversion_df["Ahorro sin Rendimiento"], label="Ahorro sin rendimiento")
            ax.set_title("Simulador de Ahorro e Inversión")
            ax.legend()
            grafico_simulacion = guardar_grafico(fig_simulador)
            st.line_chart(inversion_df, use_container_width=True)

            # Mostrar el valor final de la inversión con rendimiento centrado y resaltado
            valor_final_inversion = patrimonio_inversion[-1]
            st.markdown(
                f"<h3 style='text-align: center; color: #4CAF50; font-size: 28px;'>"
                f"Valor final estimado de la inversión después de {horizonte_inversion} años: ${valor_final_inversion:,.2f}"
                f"</h3>",
                unsafe_allow_html=True
            )

            # Mostrar el valor final de ahorro sin rendimiento centrado y resaltado
            valor_final_ahorro = patrimonio_ahorro[-1]
            st.markdown(
                f"<h3 style='text-align: center; color: #FF5722; font-size: 28px;'>"
                f"Valor acumulado sin rendimiento después de {horizonte_inversion} años: ${valor_final_ahorro:,.2f}"
                f"</h3>",
                unsafe_allow_html=True
            )

            # Generación del PDF
            if st.button("Descargar resumen en PDF"):
                datos_personales = {
                    "Nombre": st.session_state["user_data"]["Nombre"],
                    "Teléfono": st.session_state["user_data"]["Teléfono"],
                    "Email": st.session_state["user_data"]["Email"],
                    "Ciudad": st.session_state["user_data"]["Ciudad"],
                    "Edad": st.session_state["user_data"]["Edad"]
                }
                simulacion_ahorro = {
                    "Aportación Inicial": f"${aportacion_inicial:,.2f}",
                    "Aportación Periódica": f"${aportacion_periodica:,.2f}",
                    "Frecuencia de Aportación": frecuencia_aportacion,
                    "Horizonte de Inversión": f"{horizonte_inversion} años"
                }
                pdf_file = generar_pdf(datos_personales, selected_etfs_for_portfolio, weights * 100, expected_return, portfolio_volatility, simulacion_ahorro, grafico_simulacion)
                
                st.download_button(label="Descargar PDF", data=pdf_file, file_name="Cotizacion_Inversion_Patrimonial.pdf", mime="application/pdf")

    st.write("Esta es una aplicación interactiva creada para simular ETFs y carteras patrimoniales.")


