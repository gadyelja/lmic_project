import streamlit as st 
import matplotlib.pyplot as plt 
import matplotlib.ticker as mticker
import pandas as pd
import statsmodels.api as sm
import numpy as np
import io

st.set_page_config(layout="wide")




st.markdown("""
    <h1 style="text-align: center;">LMIC countries : analysis</h1>
""", unsafe_allow_html=True) # Título
path = 'list_of_countries_index.xlsx
# Importar librerias
population = pd.read_excel(path, sheet_name=8)
lmic = pd.read_excel(path, sheet_name= 9)
mark_cap = pd.read_excel(path, sheet_name= 7)
returns = pd.read_excel(path, sheet_name = 3 )
famafrench = pd.read_excel(path, sheet_name=  10)
msci = pd.read_excel(path, sheet_name=  11)
#returns = returns.iloc[:, :-2]


# Filtrar por paises LMIC
lmic = lmic[lmic["LMIC"].isin(["L", "UM", "LM"])]
population = population[population["LMIC"].isin(["L", "UM", "LM"])]
mark_cap = mark_cap[mark_cap["LMIC"].isin(["L", "UM", "LM"])]
returns = returns[returns["LMIC"].isin(["L", "UM", "LM"])]


# Ver paises vacios en returns y mark_cap
mark_cap.columns = mark_cap.columns.astype(str)
empty_mark = mark_cap[mark_cap['2015-02-27 00:00:00'].isna()] # market cap
empty_mark.columns = empty_mark.columns.astype(str)
empty_row_mark = empty_mark['Country'].count()

empty_ret = returns.loc[returns.iloc[:, 3:-1].isna().all(axis=1), [returns.columns[0]]].dropna() #returns
empty_rows = empty_ret['Country'].size

mark_cap_full = mark_cap.dropna(subset=['2015-02-27 00:00:00'])
returns_full = returns.dropna(subset=returns.columns[3:-1], how = 'all')

# Filtrar bases
common_countries = set(mark_cap_full['Country']).intersection(set(returns_full['Country']))

mark_cap_full1 = mark_cap_full[mark_cap_full['Country'].isin(common_countries)]
returns_full1 = returns_full[returns_full['Country'].isin(common_countries)]

st.subheader('Datasets: ')

col1, col2 = st.columns([1, 1]) # Organizar visualizacion


with col1:
    st.write('Population:')
    st.dataframe(population, use_container_width=True)
    
    st.write('LMIC list')
    st.dataframe(lmic, use_container_width= True)
    
    st.subheader('Countries with no returns: ')
    st.dataframe(empty_ret, use_container_width=True)
    st.write('Empty countries: ',empty_rows)
    
with col2:
    st.write('market cap')
    st.dataframe(mark_cap_full, use_container_width=True)
    
    st.write('Returns')
    st.dataframe(returns_full, use_container_width=True)
    
    st.subheader('Countries with no Market Cap: ')
    st.dataframe(empty_mark, use_container_width=True)
    st.write('Empty countries:', empty_row_mark)


st.write('Countries with market cap info:',mark_cap_full['Country'].count())
st.write('Countries with returns info: ',returns_full['Country'].count())

#################################################################################################
st.subheader('Final datasets')
col1, col2 = st.columns([1, 1])

with col1:
    st.write('Market cap per country: ')
    st.dataframe(mark_cap_full1, use_container_width=True)
    st.write('Number of countries: ', mark_cap_full1['Country'].count())

with col2:
    st.write('Returns per country: ')
    st.dataframe(returns_full1, use_container_width=True)
    st.write('Number of countrues: ',returns_full1['Country'].count())


################################################################################################
st.subheader('Statisitics:')

# Organizar df de poblacion
common_countries1 = set(population['Country']).intersection(set(returns_full1['Country']))
population1 = population[population['Country'].isin(common_countries1)]


# Filtrar paises
population1.columns = population1.columns.astype(str) # Normalizar nombres de columnas
population1 = population1.iloc[:, 0:2] # Dejar solo la primera columna

statistics_pop = population1.describe()

col1, col2 = st.columns([1, 1])

with col1:
    st.write('Population filtered: ')
    st.dataframe(population1, use_container_width=True)
    st.write('Number of countries: ',population1['Country'].size)
    
with col2:
    st.write('Statistics of population')
    st.dataframe(statistics_pop, use_container_width=True)
    
    
# Grafico: histograma de la distribucion de la poblacion 
fig, ax = plt.subplots(figsize = (4,3))
ax.hist(population1['2015-02-27 00:00:00'].dropna(), bins=80, edgecolor="black")

ax.set_xlim(0, 1500000000) # Eje X
ax.set_ylim(0, 10) # Eje Y
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax.set_xlabel("Population in 2015") # Tiulos
ax.set_ylabel("Frequency")
ax.set_title("Distribution of population in 2015 for LMIC")
plt.xticks(rotation=45)

# Mostrar gráfico en Streamlit
st.subheader('Histogram of population for countries sample')
col1, col2 = st.columns([1,1]) 

# Cortes de poblacion 
# Rangos
bins = [0,5000000, 10000000, 20000000, 30000000, 40000000, 50000000, 60000000, 70000000, 80000000, 90000000, 
        100000000, 150000000, 200000000, 300000000,  500000000, 1500000000]

# Etiquetas
labels = ['<5M','5m-10M', '10M-20M', '20M-30M', '30M-40M', '40M-50M', '50M-60M', '60M-70M', '70M-80M', '80M-90M', '90M-100M',
        '100M-150M', '150M-200M', '200M-300M', '300M-500M', '500M-1500M']


# Contar los rangos
population2 = population1['2015-02-27 00:00:00'] # Extrater columna de poblacion
population2['range'] = pd.cut(population2, bins = bins, labels = labels, right = False)
conteo = population2['range'].value_counts().sort_index() 

# Crear subgrupos
population1['2015-02-27 00:00:00'] = pd.to_numeric(population1['2015-02-27 00:00:00'], errors='coerce') # Volver numerico

# Menos de 5 millones
group1 = population1[population1['2015-02-27 00:00:00']  < 19000000]

# Entre 5 y 15 millones
group2 = population1[(population1['2015-02-27 00:00:00'] >= 19000000) & (population1['2015-02-27 00:00:00'] <= 80000000)] 

# Mas de 15 millones
group3 = population1[population1['2015-02-27 00:00:00'] > 80000000] 


# Mostrar resultados: 
with col1:
    st.pyplot(fig) # Histograma 

with col2: 
    st.write('Number of countries per range of population: ')
    st.dataframe(conteo, use_container_width=True)
    st.write('Group 1 (<19M): ', group1['2015-02-27 00:00:00'].count())
    st.write('Group 2 (19M-80M): ', group2['2015-02-27 00:00:00'].count())
    st.write('Group 3 (>80M): ', group3['2015-02-27 00:00:00'].count())
    
# Mostrar paises pertencientes a cada grupo
col1, col2, col3 = st.columns([3,3,3])

with col1:
    st.write('Countries of group 1: ')
    st.dataframe(group1, use_container_width=True)

with col2:
    st.write('Countries of group 2: ')
    st.dataframe(group2, use_container_width=True)

with col3:
    st.write('Countries of group 3: ')
    st.dataframe(group3, use_container_width=True)
    
####################################################################################################
# CAPM MODELS    
st.markdown("""
    <h1 style="text-align: center;">CAPM MODELS</h1>
""", unsafe_allow_html=True) 

## Modelo simple

# Filtrar paises por grupos

# Retornos
return_1 = returns_full1[returns_full1['Country'].isin(group1['Country'])]
return_2 = returns_full1[returns_full1['Country'].isin(group2['Country'])]
return_3 = returns_full1[returns_full1['Country'].isin(group3['Country'])]

# Market cap
mark_cap_1 = mark_cap_full1[mark_cap_full1['Country'].isin(group1['Country'])]
mark_cap_2 = mark_cap_full1[mark_cap_full1['Country'].isin(group2['Country'])]
mark_cap_3 = mark_cap_full1[mark_cap_full1['Country'].isin(group3['Country'])]

# Promedio de retornos por mes
return_1_mean = return_1.iloc[:, 3:-1].mean()
return_2_mean = return_2.iloc[:, 3:-1].mean()
return_3_mean = return_3.iloc[:, 3:-1].mean()

return_1_mean = return_1_mean.to_frame(name = 'Returns')
return_2_mean = return_2_mean.to_frame(name = 'Returns')
return_3_mean = return_3_mean.to_frame(name = 'Returns')

famafrench['RF'] = pd.to_numeric(famafrench['RF'], errors='coerce')
famafrench['Mkt-RF'] = pd.to_numeric(famafrench['Mkt-RF'], errors='coerce')# Volver a numero
famafrench.set_index('date', inplace=True)

# Ri - RF
return1_rf = (return_1_mean['Returns'] - famafrench['RF']).to_frame(name = 'Ri - Rf')
return2_rf = (return_2_mean['Returns'] - famafrench['RF']).to_frame(name = 'Ri - Rf')
return3_rf = (return_3_mean['Returns'] - famafrench['RF']).to_frame(name = 'Ri - Rf')



with col1:
    st.write('Returns of Group 1: ')
    st.dataframe(return_1, use_container_width=True)
    st.write('Market Cap of group 1: ')
    st.dataframe(mark_cap_1, use_container_width=True)
    
    st.write('Monthly mean of returns of group 1: ')
    st.dataframe(return_1_mean, use_container_width=True)
    
    st.write('Ri - Rf of group 1:')
    st.dataframe(return1_rf, use_container_width=True)

with col2:
    st.write('Returns of Group 2: ')
    st.dataframe(return_2, use_container_width=True)
    st.write('Market Cap of group 2: ')
    st.dataframe(mark_cap_2, use_container_width=True)
    
    st.write('Monthly mean of returns of group 2: ')
    st.dataframe(return_2_mean, use_container_width=True)
    
    st.write('Ri - Rf of group 2:')
    st.dataframe(return2_rf, use_container_width=True)
    
with col3:
    st.write('Returns of Group 3: ')
    st.dataframe(return_3, use_container_width=True)
    st.write('Market Cap of group 3: ')
    st.dataframe(mark_cap_3, use_container_width=True)
    
    st.write('Monthly mean of returns of group 3: ')
    st.dataframe(return_3_mean, use_container_width=True)
    
    st.write('Ri - Rf of group 3:')
    st.dataframe(return3_rf, use_container_width=True)
    
    
# resultados
st.markdown("""
    <h1 style="text-align: center;">SIMPLE MODEL NON WEIGHTED</h1>
""", unsafe_allow_html=True) 


# Modelo simple non weighted
Y_1 = return1_rf
X = famafrench['Mkt-RF']
X = sm.add_constant(X)  
modelo = sm.OLS(Y_1, X).fit()

Y_2 = return2_rf
modelo1 = sm.OLS(Y_2, X).fit()

Y_3 = return3_rf
modelo2 = sm.OLS(Y_3, X).fit()


col1, col2, col3 = st.columns([1, 1, 1])


with col1: 
    st.write('Simple CAPM Non-Weighted for group 1:')
    st.text(modelo.summary())
    
with col2: 
    st.write('Simple CAPM Non-Weighted for group 2:')
    st.text(modelo1.summary())

with col3: 
    st.write('Simple CAPM Non-Weighted for group 3:')
    st.text(modelo2.summary())
    
    
# Modelo simple weighted
st.markdown("""
    <h1 style="text-align: center;">SIMPLE MODEL WEIGHTED</h1>
""", unsafe_allow_html=True) 

# Grupo 1
weight1 = mark_cap_1['2015-02-27 00:00:00'] / mark_cap_1['2015-02-27 00:00:00'].sum()
return_1 = return_1.fillna(0)
weighted1_returns = (np.dot(return_1.iloc[:, 3:-1].T, weight1))
weighted1_returns = pd.DataFrame(weighted1_returns, columns = ['Weighted returns'])

famafrench = famafrench.reset_index()
return4_rf = (weighted1_returns['Weighted returns'] - famafrench['RF']).to_frame(name = 'WRi - Rf')

# Grupo 2
weight2 = mark_cap_2['2015-02-27 00:00:00'] / mark_cap_2['2015-02-27 00:00:00'].sum()
return_2 = return_2.fillna(0)
weighted2_returns = (np.dot(return_2.iloc[:, 3:-1].T, weight2))
weighted2_returns = pd.DataFrame(weighted2_returns, columns = ['Weighted returns'])

return5_rf = (weighted2_returns['Weighted returns'] - famafrench['RF']).to_frame(name = 'WRi - Rf')

# Grupo 3
weight3 = mark_cap_3['2015-02-27 00:00:00'] / mark_cap_3['2015-02-27 00:00:00'].sum()
return_3 = return_3.fillna(0)
weighted3_returns = (np.dot(return_3.iloc[:, 3:-1].T, weight3))
weighted3_returns = pd.DataFrame(weighted3_returns, columns = ['Weighted returns'])

return6_rf = (weighted3_returns['Weighted returns'] - famafrench['RF']).to_frame(name = 'WRi - Rf')

# Modelos simples weighted
# modelo 1
Y_4 = return4_rf
X = famafrench['Mkt-RF']
X = sm.add_constant(X)  
modelo3 = sm.OLS(Y_4, X).fit()

# Modelo 2
Y_5 = return5_rf
modelo4 = sm.OLS(Y_5, X).fit()

# Modelo 3
Y_6 = return6_rf
modelo5 = sm.OLS(Y_6, X).fit()

col1, col2, col3 = st.columns([1,1,1])
with col1: 
    st.write('Weighted returns for group 1:')
    st.dataframe(weighted1_returns, use_container_width=True)
    
    st.write('WRi - Rf for group 1:')
    st.dataframe(return4_rf, use_container_width=True)
    
    st.write('Weighted CAPM for group 1:')
    st.text(modelo3.summary())

with col2:
    st.write('Weighted returns for group 2:')
    st.dataframe(weighted2_returns, use_container_width=True)
    
    st.write('WRi - Rf for group 2:')
    st.dataframe(return5_rf, use_container_width=True)
    
    st.write('Weighted CAPM for group 2:')
    st.text(modelo4.summary())
    
with col3:
    st.write('Weighted returns for group 3:')
    st.dataframe(weighted3_returns, use_container_width=True)
    
    st.write('WRi - Rf for group 3:')
    st.dataframe(return6_rf, use_container_width=True)
    
    st.write('Weighted CAPM for group 3:')
    st.text(modelo5.summary())

######################################################################################
# Mutifactorial CAPM Non weighted
st.markdown("""
    <h1 style="text-align: center;">MULTIFACTORIAL MODEL NON WEIGHTED</h1>
""", unsafe_allow_html=True) 



# Organizar dataframes
return_1_mean = return_1_mean.reset_index().iloc[:, 1:]
return1_rf = return1_rf.reset_index()

return_2_mean = return_2_mean.reset_index().iloc[:, 1:]
return2_rf = return2_rf.reset_index()

return_3_mean = return_3_mean.reset_index().iloc[:, 1:]
return3_rf = return3_rf.reset_index()

# Modelo 1
Y_7 = return1_rf.iloc[:, 1:]
X_11 = msci.T.reset_index().iloc[:, 1:]
X_1 = pd.concat([famafrench['Mkt-RF'], X_11], axis = 1)
X_1 = sm.add_constant(X_1)
st.write(X_1)

model7 = sm.OLS(Y_7, X_1).fit()

# modelo 2
Y_8 = return2_rf.iloc[:, 1:]
X_22 = msci.T.reset_index().iloc[:, 1:]
X_2 = pd.concat([famafrench['Mkt-RF'], X_22], axis = 1)
X_2 = sm.add_constant(X_2)
model8 = sm.OLS(Y_8, X_2).fit()


# modelo 3
Y_9 = return3_rf.iloc[:, 1:]
X_33 = msci.T.reset_index().iloc[:, 1:]
X_3 = pd.concat([famafrench['Mkt-RF'], X_33], axis = 1)
X_3 = sm.add_constant(X_3)
model9 = sm.OLS(Y_9, X_3).fit()

# Mostrar resultados
col1, col2, col3 = st.columns([1,1,1])

with col1:
    st.write('Multifactorial model for group 1')
    st.text(model7.summary())
    
with col2:
    st.write('Multifactorial model for group 2: ')
    st.text(model8.summary())

with col3:
    st.write('Multifactorial model for group 3: ')
    st.text(model9.summary())


##########################################################################################################
# Multifactorial model Weighted
st.markdown("""
    <h1 style="text-align: center;">MULTIFACTORIAL MODEL WEIGHTED</h1>
""", unsafe_allow_html=True) 



# Modelo 1
Y_10 = return4_rf
X_44 = msci.T.reset_index().iloc[:, 1:]
X_4 = pd.concat([famafrench['Mkt-RF'], X_44], axis = 1)
X_4 = sm.add_constant(X_4)
model10 = sm.OLS(Y_10, X_4).fit()


# Modelo 2
Y_11 = return5_rf
X_55 = msci.T.reset_index().iloc[:, 1:]
X_5 = pd.concat([famafrench['Mkt-RF'], X_55], axis = 1)
X_5 = sm.add_constant(X_5)
model11 = sm.OLS(Y_11, X_5).fit()


# Modelo 3
Y_12 = return6_rf
X_66 = msci.T.reset_index().iloc[:, 1:]
X_6 = pd.concat([famafrench['Mkt-RF'], X_66], axis = 1)
X_6 = sm.add_constant(X_6)
model12 = sm.OLS(Y_12, X_6).fit()

col1, col2, col3 = st.columns([1,1,1])

with col1:
    st.write('Multifacotrial model 1 weighted')
    st.text(model10.summary())
    st.write(np.percentile(weighted1_returns,99))
    st.write(weighted1_returns.describe())

with col2:
    st.write('Multifacotrial model 2 weighted')
    st.text(model11.summary())
    st.write(np.percentile(weighted2_returns,99))
    st.write(weighted2_returns.describe())
    

with col3:
    st.write('Multifacotrial model 3 weighted')
    st.text(model12.summary())
    st.write(np.percentile(weighted3_returns,99))
    st.write(weighted3_returns.describe())
    





###################################################################################################

# Boton de descarga

# funcion para decargar modelos
def extract_model_summary(model):
    """Extrae coeficientes, errores estándar, p-valores, R² y otras métricas."""
    summary_df = pd.DataFrame({
        "Coefficients": model.params,
        "Standard Errors": model.bse,
        "p-values": model.pvalues,
        "Lower range 95%": model.conf_int()[0],
        "Upper range 95%": model.conf_int()[1]
    })
    summary_df.loc["R²", "Coefficients"] = model.rsquared
    summary_df.loc["R² Adjusted", "Coefficients"] = model.rsquared_adj
    summary_df.loc["F-Statistical", "Coefficients"] = model.fvalue
    return summary_df

    


st.title("📊 Download data and models")

# Check box
download_i_dfs = st.checkbox("Download initial datasets")
download_dfs = st.checkbox("Download final datasets")
download_models = st.checkbox("Download models")

if st.button("Download in Excel"):
    buffer = io.BytesIO()
    
    
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            # Guardar DataFrames seleccionados
            if download_i_dfs:
                population.to_excel(writer, sheet_name="Population", index=False)
                lmic.to_excel(writer, sheet_name="LMIC List", index=False)
                mark_cap.to_excel(writer, sheet_name="Market Cap", index=False)
                returns.to_excel(writer, sheet_name="Returns", index=False)
            
            if download_dfs:    
                mark_cap_full.to_excel(writer, sheet_name="Final Market Cap", index=False)
                returns_full.to_excel(writer, sheet_name="Final Returns", index=False)
                population1.to_excel(writer, sheet_name="Final population", index=False)
                
                return_1.to_excel(writer, sheet_name = 'ret by groups', index = False, startrow = 0)
                return_2.to_excel(writer, sheet_name = 'ret by groups', index = False, startrow = (return_1['Country'].count() + 2))
                return_3.to_excel(writer, sheet_name = 'ret by groups', index = False, startrow = (return_2['Country'].count() + 13))
                
                mark_cap_1.to_excel(writer, sheet_name = 'Mark Cap by groups', index = False, startrow = 0)
                mark_cap_2.to_excel(writer, sheet_name = 'Mark Cap by groups', index = False, startrow = (mark_cap_1['Country'].count() + 2))
                mark_cap_3.to_excel(writer, sheet_name = 'Mark Cap by groups', index = False, startrow = (mark_cap_2['Country'].count() + 13))
                
                return1_rf.to_excel(writer, sheet_name = 'Ri - Rf', index = False, startcol = 0)
                return2_rf.to_excel(writer, sheet_name = 'Ri - Rf', index = False, startcol = 3)
                return3_rf.to_excel(writer, sheet_name = 'Ri - Rf', index = False, startcol = 6)
                
                return_1_mean.to_excel(writer, sheet_name = 'Mean of returns', startcol = 0)
                return_2_mean.to_excel(writer, sheet_name = 'Mean of returns', startcol = 3)
                return_3_mean.to_excel(writer, sheet_name = 'Mean of returns', startcol = 6)
                
                weighted1_returns.to_excel(writer, sheet_name = 'Weighted returns', startcol= 0)
                weighted2_returns.to_excel(writer, sheet_name = 'Weighted returns', startcol= 3)
                weighted3_returns.to_excel(writer, sheet_name = 'Weighted returns', startcol= 6)
                
                return4_rf.to_excel(writer, sheet_name = 'WRi - Rf', index = False, startcol = 0)
                return5_rf.to_excel(writer, sheet_name = 'WRi - Rf', index = False, startcol = 3)
                return6_rf.to_excel(writer, sheet_name = 'WRi - Rf', index = False, startcol = 6)


            # Guardar estadísticas de regresiones
            if download_models:
                extract_model_summary(modelo).to_excel(writer, sheet_name="Simple model non weighted 1")
                extract_model_summary(modelo1).to_excel(writer, sheet_name="Simple model non weighted 2")
                extract_model_summary(modelo2).to_excel(writer, sheet_name="Simple model non weighted 3")
                extract_model_summary(modelo3).to_excel(writer, sheet_name="Simple model weighted 1")
                extract_model_summary(modelo4).to_excel(writer, sheet_name="Simple model weighted 2")
                extract_model_summary(modelo5).to_excel(writer, sheet_name="Simple model weighted 3")
                extract_model_summary(model7).to_excel(writer, sheet_name="Multi model non weighted 1")
                extract_model_summary(model8).to_excel(writer, sheet_name="Multi model non weighted 2")
                extract_model_summary(model9).to_excel(writer, sheet_name="Multi model non weighted 3")
                extract_model_summary(model10).to_excel(writer, sheet_name="Multi model weighted 1")
                extract_model_summary(model11).to_excel(writer, sheet_name="Multi model weighted 2")
                extract_model_summary(model12).to_excel(writer, sheet_name="Multi model weighted 3")
                

    buffer.seek(0)
    st.download_button(
        label="📥 Confirm downloading",
        data=buffer,
        file_name="LMIC_analysis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )    
    
# python -m streamlit run "C:\Users\gadyh\OneDrive\Documentos\UNISABANA\capm_model.py"
