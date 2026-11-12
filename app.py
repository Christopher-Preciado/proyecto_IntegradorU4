import streamlit as st
import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración principal de la página (Fase 5: Maquetación Web)
st.set_page_config(
    page_title="Optimización Agrícola - Proyecto Integrador U4",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Optimización Inferencial del Rendimiento Agrícola y Control de Riego")
st.subheader("Monitoreo de Telemetría en Invernaderos Inteligentes")
st.caption("TecNM Campus Cuautla | Análisis y Exploración de Datos (INC-2505)")

# ==============================================================================
# FASE 1: PREPARACIÓN DEL ENTORNO E INGESTA DETERMINISTA DE DATOS (DUCKDB)
# ==============================================================================
@st.cache_data
def generar_telemetria_agricola():
    con = duckdb.connect(database=':memory:')
    
    # Paso 1.1: Fijar la semilla determinista en DuckDB (0.42)
    con.execute("SELECT setseed(0.42);")
    
    # Paso 1.2: Generar la tabla de N=100 parcelas con las ecuaciones del proyecto
    con.execute("""
    CREATE TABLE telemetria_parcelas AS
    SELECT
        parcela_id,
        -- Predictor 1: Volumen de Riego (100.0 a 300.0 m3/ha)
        ROUND(100.0 + (random() * 200.0), 2) AS volumen_riego,
        
        -- Predictor 2: Fertilizante Aplicado (50.0 a 150.0 kg/ha)
        ROUND(50.0 + (random() * 100.0), 2) AS fertilizante_kg,
        
        -- Grupo de Prueba: Tratamiento_A (Estándar) vs Tratamiento_B (Bio-estimulante)
        CASE 
            WHEN random() < 0.50 THEN 'Tratamiento_A'
            ELSE 'Tratamiento_B'
        END AS tipo_tratamiento,
        
        -- Variable de Respuesta (rendimiento_ton):
        -- Rendimiento = Beta0 + (Beta1 * Riego) + (Beta2 * Fertilizante) + Efecto_Tratamiento + Ruido
        -- Beta0 = 5.0, Beta1 = 0.035, Beta2 = 0.020, Tratamiento_B = +1.8, Ruido = [-0.8, +0.8]
        ROUND(
            5.0 
            + (0.035 * (100.0 + (random() * 200.0)))
            + (0.020 * (50.0 + (random() * 100.0)))
            + (CASE WHEN random() < 0.50 THEN 0.0 ELSE 1.8 END)
            + (random() * 1.6 - 0.8), 2
        ) AS rendimiento_ton
    FROM generate_series(1, 100) AS t(parcela_id);
    """)
    
    # Extraer a DataFrame de Pandas
    return con.execute("SELECT * FROM telemetria_parcelas").df()

df_campo = generar_telemetria_agricola()

# ==============================================================================
# FILTROS E INTERFAZ LATERAL (FASE 5)
# ==============================================================================
st.sidebar.header("🕹️ Filtros de Control de Campo")
tratamiento_seleccionado = st.sidebar.multiselect(
    "Filtrar por Tipo de Tratamiento:",
    options=['Tratamiento_A', 'Tratamiento_B'],
    default=['Tratamiento_A', 'Tratamiento_B']
)

# Filtrar DataFrame según selección
df_filtrado = df_campo[df_campo['tipo_tratamiento'].isin(tratamiento_seleccionado)]

mostrar_tabla = st.sidebar.checkbox("Mostrar Datos de Telemetría (DuckDB)", value=False)
if mostrar_tabla:
    st.markdown("### 📋 Muestra de Telemetría Agroindustrial (DuckDB)")
    st.dataframe(df_filtrado.head(10), use_container_width=True)

# ==============================================================================
# FASE 2: ANÁLISIS DE REGRESIÓN LINEAL Y CORRELACIÓN (SUBTEMA 4.2)
# ==============================================================================
# Matriz de Correlación de Pearson (r)
num_cols = ['volumen_riego', 'fertilizante_kg', 'rendimiento_ton']
matriz_correlacion = df_campo[num_cols].corr(method='pearson')

# Variables Predictoras (X) y Respuesta (Y)
X = df_campo[['volumen_riego', 'fertilizante_kg']]
Y = df_campo['rendimiento_ton']

X_const = sm.add_constant(X)
modelo_ols = sm.OLS(Y, X_const).fit()
residuos = modelo_ols.resid

model_sk = LinearRegression()
model_sk.fit(X, Y)
y_pred = model_sk.predict(X)

r2_adj = modelo_ols.rsquared_adj
rmse = np.sqrt(mean_squared_error(Y, y_pred))

# ==============================================================================
# FASE 3: INFERENCIA ESTADÍSTICA DE TRATAMIENTOS (SUBTEMA 4.3)
# ==============================================================================
grupo_a = df_campo[df_campo['tipo_tratamiento'] == 'Tratamiento_A']['rendimiento_ton']
grupo_b = df_campo[df_campo['tipo_tratamiento'] == 'Tratamiento_B']['rendimiento_ton']

# Prueba t-Student para 2 muestras independientes (una cola: B > A)
res_ttest = stats.ttest_ind(grupo_b, grupo_a, alternative='greater')
p_valor_t = res_ttest.pvalue
t_stat = res_ttest.statistic

# Intervalo de confianza al 95% para la diferencia de medias (mu_B - mu_A)
diff_media = grupo_b.mean() - grupo_a.mean()
n_a, n_b = len(grupo_a), len(grupo_b)
var_a, var_b = grupo_a.var(ddof=1), grupo_b.var(ddof=1)
se_diff = np.sqrt((var_a / n_a) + (var_b / n_b))
ic_inferior, ic_superior = stats.t.interval(0.95, df=n_a+n_b-2, loc=diff_media, scale=se_diff)

# ==============================================================================
# FASE 5: TARJETAS KPI DE RESUMEN
# ==============================================================================
st.markdown("---")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

kpi1.metric("Rendimiento Promedio", f"{df_filtrado['rendimiento_ton'].mean():.2f} t/ha")
kpi2.metric("R² Ajustado Modelo OLS", f"{r2_adj * 100:.2f}%")
kpi3.metric("P-Valor Prueba t (B vs A)", f"{p_valor_t:.4e}")
kpi4.metric("Diferencia Medias (B-A)", f"+{diff_media:.2f} t/ha")

# Resumen de Inferencia en Expander
with st.expander("📊 Ver Resultados Detallados de Inferencia Inferencial y OLS"):
    st.write(f"**Prueba t-Student:** Estadístico t = {t_stat:.4f}, valor-p = {p_valor_t:.6f}")
    st.write(f"**Intervalo de Confianza del 95% para (μ_B - μ_A):** [{ic_inferior:.2f}, {ic_superior:.2f}] t/ha")
    if p_valor_t < 0.05:
        st.success("✅ DECISIÓN TÉCNICA: Se RECHAZA la Hipótesis Nula (H0). El Tratamiento B (Bio-estimulante) incrementa de forma estadísticamente significativa el rendimiento promedio respecto al estándar (p < 0.05).")
    else:
        st.warning("❌ DECISIÓN TÉCNICA: No se rechaza H0. No hay evidencia suficiente de mejora con el Tratamiento B.")
    
    st.text(modelo_ols.summary())

# ==============================================================================
# FASE 4: DESARROLLO DE VISUALIZACIONES EN CUADRÍCULA 2x2
# ==============================================================================
st.markdown("---")
st.markdown("### 📈 Panel de Diagnóstico e Inferencia Agrónoma")

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Mapa de Calor de Correlación de Pearson
sns.heatmap(matriz_correlacion, annot=True, fmt=".3f", cmap="YlGnBu", cbar=True, ax=axes[0, 0])
axes[0, 0].set_title('1. Matriz de Correlación de Pearson (r)', fontsize=12, fontweight='bold')

# 2. Scatter Plot: Riego vs Rendimiento con Recta OLS
sns.regplot(data=df_filtrado, x='volumen_riego', y='rendimiento_ton', color='forestgreen',
            ax=axes[0, 1], scatter_kws={'alpha': 0.7}, line_kws={'color': 'darkred', 'linewidth': 2})
axes[0, 1].set_title('2. Tendencia: Volumen de Riego vs Rendimiento', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Volumen de Riego (m³/ha)')
axes[0, 1].set_ylabel('Rendimiento (t/ha)')

# 3. Boxplot Comparativo de Tratamientos (Subtema 4.3)
sns.boxplot(data=df_campo, x='tipo_tratamiento', y='rendimiento_ton', palette=['#3498db', '#2ecc71'], ax=axes[1, 0])
sns.stripplot(data=df_campo, x='tipo_tratamiento', y='rendimiento_ton', color='black', alpha=0.3, jitter=0.2, ax=axes[1, 0])
axes[1, 0].set_title('3. Comparativa Inferencial: Tratamiento A vs B', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Tratamiento Insumo')
axes[1, 0].set_ylabel('Rendimiento (t/ha)')

# 4. Histograma y Densidad de Residuos (Diagnóstico de Normativa)
sns.histplot(residuos, kde=True, color='teal', ax=axes[1, 1], stat="density", bins=15)
axes[1, 1].set_title('4. Diagnóstico de Errores: Distribución de Residuos', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Residuo (e = Y - Ŷ)')
axes[1, 1].set_ylabel('Densidad')

plt.tight_layout()
st.pyplot(fig)