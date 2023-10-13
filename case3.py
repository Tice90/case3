import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
import plotly.figure_factory as ff
import geopandas as gpd
import folium
from folium.plugins import HeatMap
import requests
import json

#--------------------------------------------------------
#RdW data
#--------------------------------------------------------
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv('elektrischvervoer_filtered.csv')
    # Code waarmee het is gefilterd
    #df = df[['Kenteken', 'Merk', 'Handelsbenaming', 'Catalogusprijs', 'Datum eerste tenaamstelling in Nederland DT']]
    df['Datum eerste tenaamstelling in Nederland DT'] = pd.to_datetime(df['Datum eerste tenaamstelling in Nederland DT'])
    df = df[df['Datum eerste tenaamstelling in Nederland DT'].dt.year > 2012]
    
    # Verschillende benamingen merken oplossen
    brand_mapping = {
        "BMW I": "BMW",
        "JAGUAR CARS": "JAGUAR",
        "TESLA MOTORS": "TESLA",
        "MICRO COMPACT CAR SMART": "SMART",
        "M.A.N.": "MAN",
        "VW": "VOLKSWAGEN",
        "VOLKSWAGEN/ZIMNY": "VOLKSWAGEN"
    }
    df['Merk'] = df['Merk'].replace(brand_mapping)
    
    # Filter merken die minder dan 30 keer voorkomen
    counts = df['Merk'].value_counts()
    frequent_brands = counts[counts >= 30].index.tolist()
    df = df[df['Merk'].isin(frequent_brands)]
    
    return df

df = load_and_preprocess_data()

#--------------------------------------------------------
#openchargemap
#--------------------------------------------------------


#--------------------------------------------------------
#laadpalendata
#--------------------------------------------------------

df3 = pd.read_csv('laadpaaldata.csv')
df3.head()

df3.info()

#datum 2018-02-29 omzetten naar NaT, 2018 geen schrikkeljaar. 
df3['Started'] = pd.to_datetime(df3['Started'], errors='coerce')
df3['Ended'] = pd.to_datetime(df3['Ended'], errors='coerce')


#'Started' en 'Ended' naar datetime omzetten
df3['Started'] = pd.to_datetime(df3['Started'])
df3['Ended'] = pd.to_datetime(df3['Ended'])

#Extra kollom maken voor het uur van de dag
df3['HourOfDay'] = df3['Started'].dt.hour

#Verwijder rijen met negatieve waarden
df3 = df3[df3['ChargeTime'] >= 0]

df3['Weekday'] = df3['Started'].dt.day_name()

df3['Weekday'].describe()

df3['TotalEnergy (kwh)'] = df3['TotalEnergy'] / 1000
df3.head()

#de snelheid wordt berekend door de totale energie te delen door de oplaadtijd
df3['ChargeSpeed'] =  df3['TotalEnergy (kwh)'] / df3['ChargeTime'] 

# niet opgeladen tijd kijken
df3['NotChargeTime'] = df3['ConnectedTime'] - df3['ChargeTime']


#----------------------------------------------------------------
#streamlit app maken 
#----------------------------------------------------------------
page = st.sidebar.selectbox("Select a Page",['Oplaadtijd Laadpalen', "Elektrische Auto's", 'O.C.M.'])

if page == 'Oplaadtijd Laadpalen':
    st.title('Laadpalen dataonderzoek')
    st.write("Op deze pagina wordt onderzoek gedaan naar de relaties tussen de oplaadtijden, oplaadsnelheden en de maximale oplaadvermogen van auto's")
    
    tab1, tab2, tab3 = st.tabs(["Oplaadtijd vs uur per dag", "Niet-opgeladen tijd vs uur per dag", "Oplaad snelheid vs Maximale vermogen"])
    with tab1:
        st.title("Onderzoek naar laadpalen")
        st.subheader("kansdichtheid tussen de oplaadtijd per uur van de dag")
        st.write("Hier wordt de kansdichtheid weergegeven voor de oplaadtijd in uren.")

        df_chargetimes = df3[df3['ChargeTime'] <= 24]

        # Assuming df is your DataFrame and 'ChargeTime' is the column of interest
        charge_times = df_chargetimes['ChargeTime']

        #Create KDE plot for ChargeTime
        fig4 = ff.create_distplot([charge_times], ['ChargeTime'], colors=['blue'])

        # Calculate mean and median
        mean_charge_time = charge_times.mean()
        median_charge_time = charge_times.median()

        # Add vertical lines for mean and median
        fig4.add_vline(x=mean_charge_time, line_dash="dash", line_color="red", annotation_text=f'Mean: {mean_charge_time:.2f} hours', annotation_position="top right")
        fig4.add_vline(x=median_charge_time, line_dash="dash", line_color="green", annotation_text=f'Median: {median_charge_time:.2f} hours', annotation_position="bottom right")

        # Update layout
        fig4.update_layout(
            title='Histogram of Charging Time',
            xaxis_title='Charging Time (hours)',
            yaxis_title='Density'
        )
        # Show the plot
        st.plotly_chart(fig4)

        st.subheader("Kansdichtheid for elke dag in de week")
        st.write("Hier wordt de kansdichtheid weergegeven voor de oplaadtijd in uren voor elke dag in de week.")  

            # Group data by 'Weekday' and get 'ChargeTime' for each group
        hist_data = []
        group_labels = []

        for day in df3['Weekday'].unique():
            charge_times_day = df3[df3['Weekday'] == day]['ChargeTime']
            if not charge_times_day.empty:
                hist_data.append(charge_times_day)
                group_labels.append(day)

        # Create distplots for each weekday if there's data
        if hist_data:
            fig = ff.create_distplot(hist_data, group_labels, bin_size=0.5, show_hist=False)

            # Update layout
            fig.update_layout(
                title='Distribution of Charging Time by Weekday',
                xaxis_title='Charging Time (hours)',
                yaxis_title='Density'
            )

            # Display the plot
            st.plotly_chart(fig)
        else:
            print('No data available to create distplots.')

    
    with tab2:
        st.title("Onderzoek naar laadpalen")
        st.subheader("kansdichtheid niet-gebruikte laadpalen")
        st.write("Hier wordt de kansdichtheid weergegeven voor de aantal uur dat er een laadpaal aangesloten is maar niet gebruikt wordt.")

        # Assuming df is your DataFrame and 'ChargeTime' is the column of interest
        df_CHT = df3[df3['NotChargeTime'] > 0]
        not_charge_times = df_CHT['NotChargeTime']

        #Create KDE plot for ChargeTime
        fig4 = ff.create_distplot([not_charge_times], ['NotChargeTime'], colors=['blue'])

        # Calculate mean and median
        mean_not_charge_time = not_charge_times.mean()
        median_not_charge_time = not_charge_times.median()

        # Add vertical lines for mean and median
        fig4.add_vline(x=mean_not_charge_time, line_dash="dash", line_color="red", annotation_text=f'Mean: {mean_not_charge_time:.2f} hours', annotation_position="top right")
        fig4.add_vline(x=median_not_charge_time, line_dash="dash", line_color="green", annotation_text=f'Median: {median_not_charge_time:.2f} hours', annotation_position="bottom right")

        # Update layout
        fig4.update_layout(
            title='Histogram of Not Charging Time',
            xaxis_title='Not Charging Time (hours)',
            yaxis_title='Density'
        )
        # Show the plot
        st.plotly_chart(fig4)

        st.subheader("Kansdichtheid for elke dag in de week")

        st.write("Hier wordt de kansdichtheid weergegeven voor de oplaadtijd in uren voor elke dag in de week.")  

            # Group data by 'Weekday' and get 'ChargeTime' for each group
        hist_data = []
        group_labels = []

        for day in df3['Weekday'].unique():
            not_charge_times_day = df3[df3['Weekday'] == day]['NotChargeTime']
            if not not_charge_times_day.empty:
                hist_data.append(not_charge_times_day)
                group_labels.append(day)

        # Create distplots for each weekday if there's data
        if hist_data:
            fig = ff.create_distplot(hist_data, group_labels, bin_size=0.5, show_hist=True)

            # Update layout
            fig.update_layout(
                title='Distribution of No Charging Time by Weekday',
                xaxis_title='No Charging Time (hours)',
                yaxis_title='Density'
            )

            # Display the plot
            st.plotly_chart(fig)
        else:
            print('No data available to create distplots.')

    with tab3:
        st.subheader("Relatie tussen max vermogen en de oplaadsnelheid in uren.")
        st.write("Hier wordt een scatter plot weergegeven van de oplaadsnelheid in uren t.o.v de maximale vermogen.")
        fig2 = px.scatter(df3, x='MaxPower', y='ChargeSpeed', color='Weekday',
                 title='Charge Speed vs Max Power with Trendline by Weekday')
        st.plotly_chart(fig2)


if page == "Elektrische Auto's":
    st.title("Analyse van de Nederlandse Markt voor Elektrische Auto's")
    st.subheader("Een overzicht van trends, prijzen en opkomende merken in de elektrische automarkt.")


    
    
    tab1, tab2,tab3 = st.tabs(["Aantal elektrsiche auto's", "Merken", "Prijsontwikkeling"])
    with tab1:
        st.subheader("Ontwikkeling van het aantal elektrische auto's in Nederland")
        st.write("De onderstaande grafieken tonen de ontwikkeling van het aantal elektrische auto's in Nederland. De eerste grafiek toont het cumulatieve aantal tenaamstellingen van elektrische auto's, wat een beeld geeft van de groei van elektrische mobiliteit in Nederland. De tweede grafiek toont het aantal nieuwe tenaamstellingen op specifieke data, waarmee je kunt zien op welke momenten er pieken of dalen zijn in de adoptie van elektrische auto's.")
    

        @st.cache_data
        def calculate_cumulative(df):
            # Groepeer de DataFrame op basis van de datum en tel het aantal rijen
            df_cumulative = df.groupby('Datum eerste tenaamstelling in Nederland DT').size().reset_index(name='Aantal')

            # Bereken de cumulatieve som
            df_cumulative['Cumulatief Aantal'] = df_cumulative['Aantal'].cumsum()

            return df_cumulative
        
        df_cumulative = calculate_cumulative(df)

        # Maak de cumulatieve grafiek
        fig_cumulative = px.line(df_cumulative, 
                                 x='Datum eerste tenaamstelling in Nederland DT', 
                                 y='Cumulatief Aantal',
                                 title='Cumulatieve Aantal Tenaamstellingen')
        st.plotly_chart(fig_cumulative)

        # Groepeer de DataFrame op basis van de datum en tel het aantal rijen
        df_count = df.groupby('Datum eerste tenaamstelling in Nederland DT').size().reset_index(name='Aantal')

        # Maak de grafiek voor het aantal tenaamstellingen op een bepaald moment
        fig_count = px.bar(df_count, 
                           x='Datum eerste tenaamstelling in Nederland DT', 
                           y='Aantal',
                           title='Aantal Tenaamstellingen op een Bepaald Moment')
        st.plotly_chart(fig_count)
        

    with tab2:
        st.subheader("Marktdynamiek en Opkomst van Elektrische Automerken in Nederland")
        st.write("Deze visualisaties laten zien welke merken van elektrische auto's frequent worden verkocht en hoe divers de Nederlandse markt momenteel is. Naast de gevestigde namen zijn er ook nieuwe spelers bijgekomen, waardoor de markt aan het veranderen is. Opvallend is de opkomst van Chinese merken, die recent op de markt zijn gekomen")

        
        # Maak een slider om een jaar te selecteren
        selected_year = st.slider("Selecteer een jaar:", min_value=2013, max_value=2023)

        # Filter de DataFrame op basis van het geselecteerde jaar
        df_filtered = df[df['Datum eerste tenaamstelling in Nederland DT'].dt.year == selected_year]

        # Bereken de top 10 meest populaire merken voor het geselecteerde jaar
        top_10_brands = df_filtered['Merk'].value_counts().head(10)

        # Maak de Plotly Express staafdiagram
        fig = px.bar(x=top_10_brands.index, y=top_10_brands.values, title=f'Top 10 Populaire Merken in {selected_year}', labels={'x': 'Merk', 'y': 'Frequentie'})
        fig.update_layout(
            xaxis_title='Merk',
            yaxis_title='Frequentie'
        )

        # Toon de Plotly Express figuur in Streamlit
        st.plotly_chart(fig)
        
        #Histogram: Aantal geregistreerde elektrische voertuigen per merk
        plt.figure(figsize=(15, 8))
        sns.countplot(data=df, x='Merk', palette="viridis", order=df['Merk'].value_counts().index)
        plt.xticks(rotation=90)
        plt.xlabel('Merk')
        plt.ylabel('Aantal')
        plt.title('Aantal geregistreerde elektrische voertuigen per merk')
        st.pyplot(plt.gcf())


        @st.cache_data
        def filter_and_group(df, option):
            df_filtered = df.dropna(subset=['Datum eerste tenaamstelling in Nederland DT', 'Merk'])

            if option == 'Chinese merken':
                chinese_brands = ['BYD', 'AIWAYS', 'NIO', 'XPENG', 'SERES']
                df_filtered = df_filtered[df_filtered['Merk'].isin(chinese_brands)]

            df_grouped = df_filtered.groupby(['Merk', 'Datum eerste tenaamstelling in Nederland DT']).size().reset_index(name='Aantal')
            df_grouped.sort_values(['Merk', 'Datum eerste tenaamstelling in Nederland DT'], inplace=True)
            df_grouped['Cumulatief_Aantal'] = df_grouped.groupby('Merk')['Aantal'].cumsum()

            return df_grouped

        def create_fig(df_grouped, title):
            fig = px.line(df_grouped, x='Datum eerste tenaamstelling in Nederland DT', y='Cumulatief_Aantal', color='Merk', title=title)
            fig.update_layout(
                xaxis_title='Datum eerste tenaamstelling in Nederland',
                yaxis_title='Cumulatief Aantal',
            )
            return fig

        # Streamlit dropdown
        option = st.selectbox(
            'Kies welke merken je wilt zien:',
            ('Alle merken', 'Chinese merken'))

        df_grouped = filter_and_group(df, option)

        if option == 'Alle merken':
            fig = create_fig(df_grouped, 'Ontwikkeling van Merken Over Tijd')
        elif option == 'Chinese merken':
            fig = create_fig(df_grouped, 'Ontwikkeling van Chinese Merken Over Tijd')

        # Toon de grafiek
        st.plotly_chart(fig)




    with tab3:
        st.subheader("Analyse van de Catalogusprijzen van Elektrische Auto's in Nederland")
        st.write("""
        op dit tabblad wordt ingegaan op de catalogusprijzen van elektrische auto's in Nederland. De boxplot toont de gehele verdeling van prijzen en helpt met het identificeren dat er een paar outliers zijn boven de €200,000. Deze zijn daarom uit de volgende analyses gefilterd om de leesbaarheid te verbeteren.

        Het histogram wijst uit dat de meerderheid van de elektrische auto's een catalogusprijs heeft tussen de €30,000 en €65,000. Daarnaast toont de lijngrafiek dat elektrische auto's gemiddeld het duurst waren in 2016, met een prijs van meer dan €75,000. We merken echter een scherpe daling in de prijzen na 2018, wat grotendeels kan worden verklaard door de introductie van het goedkopere Tesla Model 3.

        Ten slotte biedt de scatterplot aanvullende inzichten. Ten eerste kun je de ontwikkeling volgen van hoe fabrikanten met de tijd verschillende modellen op de markt hebben gebracht, elk in een duidelijk verschillende prijsklasse. Ten tweede zien we dat auto-registraties doorgaans in cycli van vier keer per jaar gebeuren, met een duidelijke toename aan het einde van elk jaar. Ten derde wordt zichtbaar dat 2019 een kantelpunt vormt in de markt: vanaf dit moment werd het aanbod in de middenprijsklasse aanzienlijk uitgebreid, waardoor consumenten niet langer alleen de keuze hadden tussen budget- en premiummodellen.
        """)


        #Boxplot: Catalogusprijs
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=df['Catalogusprijs'])
        plt.title('Boxplot van Catalogusprijs')
        st.pyplot(plt.gcf())

        #Histogram: Gefilterde Catalogusprijs
        # Verwijder de auto's die meer kosten dan 200000
        df_filt_prijs = df[df['Catalogusprijs'] <= 200000]

        # Maak de Plotly Express histogram
        fig = px.histogram(df_filt_prijs, x='Catalogusprijs', title=f'Histogram van Catalogusprijs')

        # Toon de Plotly Express figuur in Streamlit
        st.plotly_chart(fig)


        # Extract year from 'datum' column
        df['Jaar'] = df['Datum eerste tenaamstelling in Nederland DT'].dt.year
        # Group the data by year and calculate the average catalog price
        average_price_by_year = df.groupby('Jaar')['Catalogusprijs'].mean()

        # Create the Plotly Express line plot
        fig = px.line(average_price_by_year, x=average_price_by_year.index, y=average_price_by_year.values, title='Gemiddelde Catalogusprijs over tijd')
        fig.update_layout(
            xaxis_title='Jaar',
            yaxis_title='Gemiddelde Catalogusprijs'
        )

        # Show the Plotly Express figure in Streamlit
        st.plotly_chart(fig)

        
        # Filter en neem de willekeurige steekproef van de DataFrame
        df_filt_prijs = df[df['Catalogusprijs'] <= 200000]

        # Haal unieke merken uit de DataFrame
        unique_brands = sorted(df['Merk'].unique().tolist())

        # Voeg 'Alle merken' toe aan de lijst
        options = ['Alle merken'] + unique_brands

        # Streamlit selectbox
        selected_brand = st.selectbox(
            'Kies welke merken je wilt zien:',
            options
        )

        # Bepaal het aantal rijen van het geselecteerde merk
        num_rows_selected_brand = len(df_filt_prijs[df_filt_prijs['Merk'] == selected_brand])

        # Pas de steekproef aan op basis van het aantal rijen
        if selected_brand != 'Alle merken' and num_rows_selected_brand < 1500:
            df_filt_prijs = df_filt_prijs[df_filt_prijs['Merk'] == selected_brand]
            sample_ratio = 1  # 1:1 steekproef
        else:
            df_filt_prijs = df_filt_prijs.sample(frac=1/15, random_state=42)
            if selected_brand != 'Alle merken':
                df_filt_prijs = df_filt_prijs[df_filt_prijs['Merk'] == selected_brand]
            sample_ratio = 15  # 1:15 steekproef

        # Maak de scatterplot met Plotly Express
        dynamic_title = f"Prijzen en registraties van een 1:{sample_ratio} willekeurige steekproef van elektrische auto's in Nederland"
        fig = px.scatter(df_filt_prijs, 
                         x='Datum eerste tenaamstelling in Nederland DT', 
                         y='Catalogusprijs',
                         color='Merk', 
                         title=dynamic_title,
                         labels={'Datum eerste toelating DT': 'Datum eerste toelating', 'Catalogusprijs': 'Catalogusprijs'},
                         hover_data=['Merk', 'Handelsbenaming', 'Catalogusprijs'],
                         )

        # Pas de grootte van de markers aan
        fig.update_traces(marker=dict(size=3))

        # Pas het lay-out aan
        fig.update_layout(
            xaxis_title="Datum eerste tenaamstelling",
            yaxis_title="Catalogusprijs",
            xaxis=dict(
                tickmode='linear',  # "linear" voor gelijk verdeelde ticks of "auto" voor automatisch
                tick0='2013-01-01',  # Startdatum voor de ticklabels
                dtick="M12"  # Frequentie van de ticks, "M12" betekent een tick elke 12 maanden
            ),
            height=600  # Hoogte van de plot
        )

        # Toon de Plotly Express figuur in Streamlit
        st.plotly_chart(fig)
        
if page == "O.C.M.":
    st.title("Open charge map api")

    st.subheader("Hieronder laden we een API in en zetten deze om naar een bruikbare Pandas DataFrame. Daarnaast maken we gebruik van GeoPandas en een zelfgemaakte Excel-sheet om de data te transformeren, waardoor we in staat zijn bepaalde conclusies te trekken.")


#     response= requests.get("https://api.openchargemap.io/v3/poi/?output=json&countrycode=NL&maxresults=10000&compact=true&verbose=false&key=4308c1bf-a620-4b40-bc6f-0bf156d35d61")


#     # In[44]:


#     responsejson = response.text



#     # In[45]:


#     d = json.loads(responsejson)
#     pd.json_normalize(d)


#     # In[46]:


#     Laadpalen = pd.json_normalize(d)
#     df4 = pd.json_normalize(Laadpalen.Connections)
#     df5 = pd.json_normalize(df4[0])
#     Laadpalen = pd.concat([Laadpalen, df5], axis=1)
#     Laadpalen.head()


#     # In[47]:


#     gdf = gpd.GeoDataFrame.from_file("provinces.geojson")


#     # In[48]:




#     # Hierboven hebben we alle data ingeladen en geanalyseerd. Ook hebben we om de data bruibaar te maken een geopanda dataframe ingeladen om zo de postcodes aan provincies te koppelen.
#     # 

#     # In[49]:


#     Laadpalen['postcode_without_letters'] = Laadpalen['AddressInfo.Postcode'].str.extract(r'(\d+)')
#     Laadpalen.head()


#     # In[50]:


#     df_post= pd.read_excel('provincie en postcodes.xlsx')


#     # In[51]:


#     df_post = pd.DataFrame(df_post)
#     df_post.head()


#     # In[52]:


#     Laadpalen = pd.merge(Laadpalen, df_post, left_on="postcode_without_letters", right_on=df_post['postcode'].astype(str), how='left')


#     # Hierna hebben we de data verder bekeken en ingezoemd op de postcodes aangezien veel andere columns een hoop na waardes hadden.



#     Laadpalen['provincie'].dropna()


#     # In[55]:


#     Laadpalen[Laadpalen['AddressInfo.Postcode'].isna()]


#     # In[56]:


#     Laadpalen['AddressInfo.Postcode'].dropna()


#     # In[57]:


#     Aantal_Laadpalen_provincie = Laadpalen.value_counts(["provincie"])
#     Aantal_Laadpalen_provincie = Aantal_Laadpalen_provincie.reset_index()
#     Aantal_Laadpalen_provincie = Aantal_Laadpalen_provincie.rename(columns={0: 'Aantal_Laadpalen'})
#     Aantal_Laadpalen_provincie = Aantal_Laadpalen_provincie.sort_values('provincie')

#     Aantal_Laadpalen_provincie['provincie'].replace(['Friesland'],'Friesland (Fryslân)', inplace=True)



#     # In[58]:


#     merged_df = Aantal_Laadpalen_provincie.merge(gdf, left_on='provincie', right_on='name')


#     # In[59]:


#     geojson = px.data.election_geojson()
#     # geojson


#     # In[60]:


#     import geopandas as gpd

#     merged_gdf = gpd.GeoDataFrame(merged_df, geometry='geometry')
#     merged_gdf.head()


#     # In[61]:


#     print(merged_gdf.columns)


#     # de geopanda dataframe samengevoegd met de laadpalen per provincie om zo een plot te kunnen laten zien van hoeveel laadpalen per provincie. hieronder dan ook een folium heatmap ter verduidelijking waar de meeste laadpalen zich bevinden. en de choropleth eronder om zo het verschil in provincie te laten zien.

#     # In[62]:


#     import folium
#     from folium.plugins import HeatMap

#     # Create a map centered around the Netherlands
#     m = folium.Map(location=[52.1326, 5.2913], zoom_start=7)

#     # Create a HeatMap layer with a different color scheme
#     heat_data = Laadpalen[['AddressInfo.Latitude', 'AddressInfo.Longitude']].dropna().values
#     HeatMap(heat_data, gradient={0.2: 'green', 0.4: 'yellow', 0.6: 'orange', 1: 'red'}, min_opacity=0.3, max_val=2000, radius=7, blur=10, max_zoom=1).add_to(m)

#     # Convert Folium map to HTML string
#     m = m._repr_html_()

#     # Use the Streamlit html component to display the Folium map
#     st.components.v1.html(m, width=800, height=600)


#     # In[63]:


#     fig = px.choropleth_mapbox(merged_gdf, geojson=gdf.geometry, locations=gdf.index, color='Aantal_Laadpalen',
#                                color_continuous_scale="Viridis",
#                                # range_color=(0, 12),
#                                mapbox_style="carto-positron",
#                                # featureidkey="properties.provincie",
#                                zoom=7, center = {"lat": 52.1326, "lon": 5.2913},
#                                opacity=0.5,
#                                labels={'unemp':'unemployment rate'}
#                               )
#     st.plotly_chart(fig)
    def get_data():
        response = requests.get("https://api.openchargemap.io/v3/poi/?output=json&countrycode=NL&maxresults=10000&compact=true&verbose=false&key=4308c1bf-a620-4b40-bc6f-0bf156d35d61")
        data = json.loads(response.text)
        return pd.json_normalize(data)

    def prepare_data(Laadpalen):
        Laadpalen['postcode_without_letters'] = Laadpalen['AddressInfo.Postcode'].str.extract(r'(\d+)')
        df_post = pd.read_excel('provincie en postcodes.xlsx')
        return pd.merge(Laadpalen, df_post, left_on="postcode_without_letters", right_on=df_post['postcode'].astype(str), how='left')

    def count_charging_stations_by_province(Laadpalen):
        counts = Laadpalen.value_counts(["provincie"])
        counts = counts.reset_index().rename(columns={0: 'Aantal_Laadpalen'})
        counts['provincie'].replace(['Friesland'],'Friesland (Fryslân)', inplace=True)
        return counts

    def render_map(Laadpalen):
        m = folium.Map(location=[52.1326, 5.2913], zoom_start=7)
        heat_data = Laadpalen[['AddressInfo.Latitude', 'AddressInfo.Longitude']].dropna().values
        HeatMap(heat_data, gradient={0.2: 'green', 0.4: 'yellow', 0.6: 'orange', 1: 'red'}, min_opacity=0.3, max_val=2000, radius=7, blur=10, max_zoom=1).add_to(m)
        return m._repr_html_()

    def main():

        Laadpalen = get_data()
        Laadpalen = prepare_data(Laadpalen)
        gdf = gpd.GeoDataFrame.from_file("provinces.geojson")

        counts = count_charging_stations_by_province(Laadpalen)
        merged_gdf = gpd.GeoDataFrame(gdf.merge(counts, left_on='name', right_on='provincie'), geometry='geometry')

        m = render_map(Laadpalen)
        st.components.v1.html(m, width=800, height=600)

        st.write("De Geopanda Dataframe samengevoegd met de laadpalen per provincie om zo met een choropleth plot te kunnen laten zien hoeveel laadpalen per provincie er zijn.")
        
        fig = px.choropleth_mapbox(merged_gdf, geojson=gdf.geometry, locations=gdf.index, color='count',
                                   color_continuous_scale="Viridis",
                                   mapbox_style="carto-positron",
                                   zoom=5, center = {"lat": 52.1326, "lon": 5.2913},
                                   opacity=0.5)
        st.plotly_chart(fig)

    if __name__ == '__main__':
        main()
