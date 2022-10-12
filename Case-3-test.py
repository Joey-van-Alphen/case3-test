#!/usr/bin/env python
# coding: utf-8

# # Case 3 - Team 12

# * Joey van Alphen
# * Mohamed Garad
# * Nusret Kaya
# * Shereen Macnack

# In[1]:


#pip install streamlit-folium
#!pip install streamlit
#!pip install statsmodels
#!pip install cbsodata


# In[2]:


import requests
import pandas as pd
import folium
import streamlit as st
from streamlit_folium import folium_static
import numpy as np
import geopandas as gpd
import plotly.express as px
import statsmodels.api as sm
import cbsodata
import matplotlib.pyplot as plt


# ## Dataset 1 Open Charge Map

# In[3]:


response = requests.get('https://api.openchargemap.io/v3/poi/?output=json&countrycode=NL&maxresults=1000&compact=true&verbose=false&key=6ba1f76e-aefd-4fca-aeea-caa80b9e24a3')
json = response.json()
df = pd.DataFrame(json)

df.head()


# In[4]:


response = requests.get('https://api.openchargemap.io/v3/poi/?output=json&countrycode=NL&maxresults=1000&compact=true&verbose=false&key=6ba1f76e-aefd-4fca-aeea-caa80b9e24a3')
responsejson = response.json()
###Dataframe bevat kolom die een list zijn.
#Met json_normalize zet je de eerste kolom om naar losse kolommen
Laadpalen = pd.json_normalize(responsejson)
#Daarna nog handmatig kijken welke kolommen over zijn in dit geval Connections
#Kijken naar eerst laadpaal op de locatie
#Kan je uitpakken middels:
df4 = pd.json_normalize(Laadpalen.Connections)
df5 = pd.json_normalize(df4[0])
df5.head()
###Bestanden samenvoegen
Laadpalen = pd.concat([Laadpalen, df5], axis=1)
Laadpalen.head()


# In[5]:


#Laadpalen.columns


# In[6]:


df2 = (Laadpalen[['AddressInfo.ID', 'AddressInfo.Title', 'AddressInfo.AddressLine1',
       'AddressInfo.Town', 'AddressInfo.StateOrProvince',
       'AddressInfo.Postcode', 'AddressInfo.CountryID','AddressInfo.Latitude', 'AddressInfo.Longitude', 'ConnectionTypeID', 'Quantity']])
df2.head(50)



# In[ ]:





# ### Dataset met geometries combineren

# In[7]:


#countries = gpd.read_file('Grenzen_van_alle_Nederlandse_gemeenten_en_provincies.geojson')
#countries.head()
geodata_url = 'https://geodata.nationaalgeoregister.nl/cbsgebiedsindelingen/wfs?request=GetFeature&service=WFS&version=2.0.0&typeName=cbs_gemeente_2017_gegeneraliseerd&outputFormat=json'
gemeentegrenzen = gpd.read_file(geodata_url)
gemeentegrenzen.head(50)


# In[8]:


aantal_per_gem = df2['AddressInfo.Town'].value_counts()
df4 = pd.DataFrame(aantal_per_gem)
df5 = df4.reset_index(level=0)
df5.columns = ['statnaam','aantal_laadpalen']
df5.head()


# In[9]:


#aantal_per_gem = df2.groupby('AddressInfo.Town')['AddressInfo.Town'].count()
#df4 = pd.DataFrame(aantal_per_gem)
#df4.columns = ['gemeente', 'aantal laadpalen']
#df4.head(50)


# In[10]:


df_samen = df5.merge(gemeentegrenzen, how='left', on='statnaam')
df_samen = df_samen[['statnaam', 'aantal_laadpalen', 'geometry']]
df_samen.head()



# In[11]:


df_samen.isna().sum()


# In[12]:


df_samen.dropna(inplace=True)


# In[13]:


df_samen.isna().sum()


# In[14]:


#df_samen.head()
#df_samen = df_samen[df_samen['GEMEENTENAAM'] != "Amsterdam"]
#df_samen.head()

geo_df_crs = {'init': 'epsg:4326'}
geo_df = gpd.GeoDataFrame(df_samen, crs= geo_df_crs, geometry = df_samen.geometry)
geo_df.head()


# In[15]:


m2 = folium.Map(location= [52.371807, 4.896029], zoom_start = 12)


# In[ ]:





# In[16]:


m2.choropleth(
    geo_data=geo_df,
    name="geometry",
    data=geo_df,
    columns= ["statnaam","aantal_laadpalen"],
    key_on="feature.properties.statnaam",
    fill_color="RdYlGn_r",
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name="Aantal laadpalen per gemeente")



# In[ ]:





# In[17]:


#df6.loc[df6['GEMEENTENAAM'] == "Almere"]


# In[18]:


df4['AddressInfo.Town'].unique()


# In[19]:


#duplicated = df.duplicated(subset = 'AddressInfo.Town', keep = False)


# In[20]:


#df2.groupby('AddressInfo.Town').count()


# In[ ]:





# In[21]:


location_select = st.sidebar.selectbox('Welk gebied wil je zien?', ('Almere', 'Utrecht', 'Amsterdam'))

if location_select == 'Almere': 
    location = [52.371807, 4.896029]
elif location_select =='Utrecht':
    location = [52.0893191, 5.1101691]
elif location_select == 'Amsterdam':
    location = [52.371807, 4.896029]


m = folium.Map(location=location, zoom_start=12)

for row in df2.iterrows():
    row_values = row[1]
    location = [row_values['AddressInfo.Latitude'], row_values['AddressInfo.Longitude']]
    popup = row_values['AddressInfo.Title']
    circle = folium.Circle(location = location, popup = popup)
    circle.add_to(m)



# In[22]:


#df2.loc[df2['AddressInfo.Town'] == 'Almere']


# In[23]:


#df2.loc[df2['AddressInfo.Town'] == na]


# ## Dataset 2: laadpaaldata

# In[24]:


df = pd.read_csv('laadpaaldata.csv')
df.head(100)


# In[25]:


df.info()


# In[26]:


df.isna().sum()


# In[27]:


#df.loc[df.duplicated(subset='Started', keep = False)]
#beide 'duplicates' hebben andere eindtijd en andere waardes, dus hier is geen sprake van duplicates


# In[28]:


df['ConnectedTime'].hist() 



# In[29]:


#outliers verwijderen
df2 = df[df['ConnectedTime'] <= 30] 
df2['ConnectedTime'].hist()


# In[30]:


df2['ChargeTime'].hist() 


# In[31]:


df3 = df2[df2['ChargeTime']>0] 
df4 = df3[df3['ChargeTime']<9]
df4['ChargeTime'].hist()


# In[32]:


df4['bezethouden'] = df4['ConnectedTime']-df4['ChargeTime']
df5 = df4[df4['bezethouden']>0]


# In[33]:


df5['bezethouden'].hist()


# In[34]:


#df5.loc[df5['ChargeTime'] <=0]


# In[35]:


fig1 = px.histogram(df5, x="bezethouden", nbins=25, title='Bezethouden van een laadpaal')

mean_bezethouden = df5['bezethouden'].mean()
median_bezethouden = df5['bezethouden'].median()
my_annotation1 = {'x': 0.1, 'y': 1.05, 'xref': 'paper', 'yref': 'paper', 'showarrow': False,"text": f"Het gemiddelde is: {mean_bezethouden}"} 
my_annotation2 = {'x': 0.8, 'y': 1.05, 'xref': 'paper', 'yref': 'paper', 'showarrow': False,"text": f"De mediaan is: {median_bezethouden}"} 
fig1.update_layout({'annotations': [my_annotation1, my_annotation2]})

fig1.update_xaxes(title_text="Het bezethouden van een laadpaal in uren")
fig1.update_yaxes(title_text="Aantal")

#fig1.show()


# In[36]:


fig2 = px.histogram(df5, x="ChargeTime", nbins= 20, title='Laadtijd van een laadpaal')

mean_chargetime = df5['ChargeTime'].mean()
median_chargetime = df5['ChargeTime'].median()
my_annotation1 = {'x': 0.1, 'y': 1.05, 'xref': 'paper', 'yref': 'paper', 'showarrow': False,"text": f"Het gemiddelde is: {mean_chargetime}"} 
my_annotation2 = {'x': 0.8, 'y': 1.05, 'xref': 'paper', 'yref': 'paper', 'showarrow': False,"text": f"De mediaan is: {median_chargetime}"} 
fig2.update_layout({'annotations': [my_annotation1, my_annotation2]})

fig2.update_xaxes(title_text="De tijd dat de laadpaal echt aan het laden is in uren")
fig2.update_yaxes(title_text="Aantal")

#fig2.show()


# In[37]:


fig3 = px.histogram(df5, x="ConnectedTime", nbins=25, title='Tijd verbonden aan een laadpaal')

mean_connectedtime = df5['ConnectedTime'].mean()
median_connectedtime = df5['ConnectedTime'].median()
my_annotation1 = {'x': 0.1, 'y': 1.05, 'xref': 'paper', 'yref': 'paper', 'showarrow': False,"text": f"Het gemiddelde is: {mean_connectedtime}"} 
my_annotation2 = {'x': 0.9, 'y': 1.05, 'xref': 'paper', 'yref': 'paper', 'showarrow': False,"text": f"De mediaan is: {median_connectedtime}"} 
fig3.update_layout({'annotations': [my_annotation1, my_annotation2]})

fig3.update_xaxes(title_text="De tijd dat de laadpaal is verbonden in uren")
fig3.update_yaxes(title_text="Aantal")

#fig3.show()


# In[38]:


fig4 = px.histogram(df5, x=['bezethouden', 'ChargeTime'], nbins=25, barmode="overlay", title='Tijd verbonden aan een laadpaal')
#fig4.show()


# In[39]:


data_select = st.sidebar.selectbox('Welke maand wil je zien?', ('Januari', 'Februari', 'Maart'))

if data_select == 'Januari': 
    data = df5[(df5['Ended']>='2018-01-01')&(df5['Ended']<='2018-01-31')].sort_values(by='Ended')
elif data_select =='Februari':
    data = df5[(df5['Ended']>='2018-02-01')&(df5['Ended']<='2018-02-31')].sort_values(by='Ended')
elif data_select == 'Maart':
    data = df5[(df5['Ended']>='2018-03-01')&(df5['Ended']<='2018-03-31')].sort_values(by='Ended')
        

fig5 = px.line(data, x='Ended', y='MaxPower')


# In[40]:


fig6 = px.scatter(df5, x='TotalEnergy', y ='MaxPower', trendline="ols")


# ## Dataset 3: Elektrische voertuigen

# In[41]:


df1 = pd.read_csv('Elektrische_voertuigen.csv', low_memory=False)
df1.head()


# In[42]:


#df1.columns


# In[43]:


#elektrische_voertuigen = df1[['Merk', 'Voertuigsoort','Handelsbenaming', 'Catalogusprijs', 'Datum eerste toelating', 'Cilinderinhoud']]
#elektrische_voertuigen_streamlit  = elektrische_voertuigen.assign(Datum_eerste_toelating = pd.to_datetime(elektrische_voertuigen['Datum eerste toelating'], format='%Y%m%d') )
#elektrische_voertuigen_streamlit.head()




# In[44]:


#elektrische_voertuigen_streamlit.to_csv(r"C:\HvA 2022-2023\Minor Data Science\Titanic Case\elektrische_voertuigen_streamlit.csv", index = False)

elektrische_voertuigen = pd.read_csv('Elektrische_voertuigen_streamlit.csv')
elektrische_voertuigen.head()


# In[45]:


elektrische_voertuigen = elektrische_voertuigen.assign(Type = np.where(elektrische_voertuigen['Cilinderinhoud'].isna(),'Electric','Hybrid'))
elektrische_voertuigen.head()


# In[46]:


#elektrische_voertuigen.loc[elektrische_voertuigen['Type'] == 'Hybrid']


# In[ ]:





# In[47]:


df_per_type = elektrische_voertuigen[['Datum_eerste_toelating', 'Type']]
df_per_type.head()


# In[48]:


df_electric = df_per_type[df_per_type['Type']=='Electric']
aantal_eper_jaar = df_electric['Datum_eerste_toelating'].value_counts()
df_e = pd.DataFrame(aantal_eper_jaar)
df_e = df.reset_index(level=0)
df_e.head()


# In[49]:


df_hybrid = df_per_type[df_per_type['Type']=='Hybrid']
aantal_hper_jaar = df_hybrid['Datum_eerste_toelating'].value_counts()
df2 = pd.DataFrame(aantal_hper_jaar)
df2 = df.reset_index(level=0)
df2.head()


# In[50]:


aantal_auto = elektrische_voertuigen['Merk'].value_counts()
df = pd.DataFrame(aantal_auto)
df.head()


# In[51]:


aantal_per_jaar = elektrische_voertuigen['Datum_eerste_toelating'].value_counts()
df = pd.DataFrame(aantal_per_jaar)
df = df.reset_index(level=0)
df= df[df['index']>'2010']
df = df.sort_values(by= 'index')
df['cumsum'] = df['Datum_eerste_toelating'].cumsum()
df.head()

#df['index'] = df['index'].dt.year
#df = df.groupby("index")['Datum_eerste_toelating'].count()
#df.head(100)


# In[52]:


fig6 = px.line(df, x='index', y ='cumsum' )
#fig.update_yaxes(type="log")
#fig.show()


# In[53]:


df_afjaar = df[(df['index']>'2021-10-1')&(df['index']<'2022-10-1')]
df_afjaar.head()
fig7 = px.line(df_afjaar, x='index', y ='cumsum' )
#fig.update_yaxes(type="log")
#fig.show()


# ## Streamlit deel

# In[54]:


st.title('Case 3 Elektrisch vervoer')


# In[55]:


folium_static(m)
folium_static(m2)
st.plotly_chart(fig1)
st.plotly_chart(fig2)
st.plotly_chart(fig3)
st.plotly_chart(fig4)
st.plotly_chart(fig5)
st.plotly_chart(fig6)

