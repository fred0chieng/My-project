import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport
import time
import os
from sklearn.metrics import classification_report,mean_squared_error
import xgboost as xgb
import sklearn.metrics as metrics
import streamlit as st
from streamlit_elements import elements, mui, html
from streamlit_option_menu import option_menu
from streamlit_extras.colored_header import colored_header
from streamlit_extras.card import card
from streamlit_oauth import OAuth2Component

# Set page config
st.set_page_config(
    page_title="Electricity Consumption Data Analysis",
    page_icon="random",
    initial_sidebar_state='expanded',
    menu_items={
        'Get Help': 'mailto:faochieng@kabarak.ac.ke',
        'report a bug': 'mailto:faochieng@kabarak.ac.ke',
        "About": "This website is created by Fredrick Ochieng showcasing a deep exploratory data analysis of three zones in Morocco",
    }
)
G_CLIENT_ID = "1080830845886-v54tcdqs3gjl0u7dpk37vsnb9jclcmu7.apps.googleusercontent.com"
G_CLIENT_SECRET = "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"
G_AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/auth"
G_TOKEN_URL = "https://oauth2.googleapis.com/token"
G_REDIRECT_URI = "https://<YOUR ADDRESS>/component/streamlit_oauth.authorize_button/index.html"  # Replace with your redirect URI
G_SCOPE = "email profile openid"  # Define the scope you need
G_oauth2 = OAuth2Component(G_CLIENT_ID, G_CLIENT_SECRET, G_AUTHORIZE_URL, G_TOKEN_URL)
google_icon_url = "https://www.gstatic.com/images/branding/product/2x/google_cloud_48dp.png"


AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
TOKEN_URL = "https://github.com/login/oauth/access_token"
REDIRECT_URI = "http://localhost:8501/"

CLIENT_ID = "Iv1.862d943ff2d822ac"
CLIENT_SECRET  = "7d2aacccc8cf688ee82afdf86c47e251370abd78"

# Create OAuth2Component instance
oauth2 = OAuth2Component(CLIENT_ID, CLIENT_SECRET, AUTHORIZE_URL, TOKEN_URL)
github_icon_url = "https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png"



@st.cache_data
def get_df():
    return load_data()

@st.cache_data
def get_profile_report(df):
    return ProfileReport(df, title="Electricity consumption Profile Report")

def create_features(df):
    """
    Create time series features based on time series index.
    """
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('powerconsumption.csv')
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)
    return df


@st.cache_data
def convert_df(df_frame):
    return df_frame.to_csv().encode('utf-8')

def create_card(image_path, title, description):
    # HTML/CSS for styling the card
    st.markdown(
        f"""
        <style>
            .card {{
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
                background-color: #f5f5f5;
                margin-bottom: 20px;
            }}
            .card-title {{
                font-size: 24px;
                font-weight: bold;
                color: #333;
                margin-bottom: 10px;
            }}
            .card-description {{
                font-size: 16px;
                color: #666;
            }}
        </style>
        <div class="card">
            <img src="{image_path}" alt="chart" style="width:100%">
            <div class="card-title">{title}</div>
            <div class="card-description">{description}</div>
        </div>
        """,
        unsafe_allow_html=True
    )




def main():
    if 'token' not in st.session_state:
        # If not, show authorize button
        result = oauth2.authorize_button("Continue with Google", G_REDIRECT_URI, G_SCOPE, icon=google_icon_url)
        if result and 'token' in result:
            # If authorization successful, save token in session state
            st.session_state.token = result.get('token')
            st.rerun()
    else:
        # If token exists in session state, show the token
        token = st.session_state['token']
        st.json(token)
        if st.button("Refresh Token"):
            # If refresh token button is clicked, refresh the token
            token = oauth2.refresh_token(token)
            st.session_state.token = token
            st.rerun()

    result = oauth2.authorize_button("Continue with GitHub", REDIRECT_URI, scope="user", icon=github_icon_url)

    if result and "token" in result:
        # Access token obtained successfully
        access_token = result["token"]
        # Use the access token for API requests
        # st.write("Access token:", access_token)
        st.session_state.logged_in = True

        # Main function to run the Streamlit app
        st.sidebar.markdown("---")
        with st.sidebar:
            menu = option_menu(
                menu_icon="cast",
                menu_title="Menu",
                options=["Data", "Analysis", "About", "Contact us"],
                icons=["database-gear", "graph-up", "info-circle", "envelope"],
                default_index=0,
            )
        df = load_data()
        df = create_features(df)

        df['SMA10'] = df['PowerConsumption_Zone1'].rolling(10).mean()
        df['SMA15'] = df['PowerConsumption_Zone1'].rolling(15).mean()
        df['SMA30'] = df['PowerConsumption_Zone1'].rolling(30).mean()

        # Extract month from Datetime column
        # df['month'] = pd.to_datetime(df['Datetime']).dt.month

        # Create a filtering interface
        selected_month = st.sidebar.selectbox('Select Month', df['month'].unique())

        # Filter the dataset based on the selected month
        filtered_df = df[df['month'] == selected_month]

        if menu == "Analysis":
            st.title("Power Consumption EDA Dashboard")
            st.markdown("---")
            col = st.columns(2)

            with col[0]:
                card(
                    title="",
                    text="Electricity Usage Predictions",
                    image="https://electricityconsumptionanalysisdata.s3.eu-north-1.amazonaws.com/predictedusage.png",
                    url="https://electricityconsumptionanalysisdata.s3.eu-north-1.amazonaws.com/predictedusage.png",
                    key="card"
                )
            with col[1]:
                card(
                    title="",
                    text="Predicted Usage against real data",
                    image="https://electricityconsumptionanalysisdata.s3.eu-north-1.amazonaws.com/actualvspredictedusage.png",
                    url="https://electricityconsumptionanalysisdata.s3.eu-north-1.amazonaws.com/actualvspredictedusage.png",
                    key="card1"
                )

            with elements("multiple_children"):

                mui.Button(
                    # mui.icon.EmojiPeople,
                    # mui.icon.DoubleArrow,
                    mui.icon.Power,

                    "Charts and Analysis"
                )

            st.subheader("Relavant Features")

            with elements("style_mui_sx"):
                mui.Box(
                    "The two most important factors are the 15-day SMA and the hour of that specific measurement. The temperature, as expected, is the only external factor that is relatively significant to predict energy consumption.",
                    sx={
                        "bgcolor": "background.paper",
                        "boxShadow": 1,
                        "borderRadius": 2,
                        "p": 2,
                        "minWidth": 100,

                    },

                )
            create_card(
                "https://electricityconsumptionanalysisdata.s3.eu-north-1.amazonaws.com/important-feature.png", "",
                "By plotting the importance each feature played in predicting the correct result we can clearly notice some relevant patterns.")

            col = st.columns(2)

            st.sidebar.subheader("More chart options")
            if st.sidebar.checkbox("Correlation Heatmap"):
                st.subheader("Correlation Heatmap")
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(df.corr(), annot=True, ax=ax, cmap='vlag', fmt='.1g',
                            annot_kws={'fontsize': 14, 'fontweight': 'regular'},
                            xticklabels=['Temp', 'Hum', 'Wind', 'Gen Diff Flows', 'Diff Flows', 'Power Z1',
                                         'Power Z2', 'Power Z3'],
                            yticklabels=['Temp', 'Hum', 'Wind', 'Gen Diff Flows', 'Diff Flows', 'Power Z1',
                                         'Power Z2', 'Power Z3'])
                plt.xticks(fontsize=11)
                plt.yticks(fontsize=11)
                st.pyplot(fig)

            if st.sidebar.checkbox("Histogram of Power Consumption"):
                fig4, ax = plt.subplots(figsize=(20, 10))

                st.subheader("Histogram of Power Consumption")
                plt.hist(df["PowerConsumption_Zone3"], bins=20)
                plt.xlabel("Power Consumption (Zone 3)")
                plt.ylabel("Frequency")
                st.pyplot(fig4)

            if st.sidebar.checkbox("All Charts"):
                fig, ax = plt.subplots(figsize=(30, 20))
                st.subheader("Power Consumption in KW against time in  the 3 Zones")

                zone1 = sns.boxplot(data=df, x='hour', y='PowerConsumption_Zone1', palette='Oranges',
                                    showfliers=False)
                zone2 = sns.boxplot(data=df, x='hour', y='PowerConsumption_Zone2', palette='Reds', showfliers=False)
                zone3 = sns.boxplot(data=df, x='hour', y='PowerConsumption_Zone3', palette='Blues',
                                    showfliers=False)

                plt.suptitle('KW by Hour', fontsize=15)
                plt.xlabel('hour', fontsize=12)
                plt.ylabel('Power Consumption in KW', fontsize=12)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                st.pyplot(fig)

                # Plot Temperature vs. Power Consumption_Zone2
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(data=filtered_df, x='Temperature', y='PowerConsumption_Zone2', ax=ax)
                plt.title('Temperature vs. Power Consumption (Zone 2)')
                st.pyplot(fig)
                # Close the Matplotlib figure to release memory
                plt.close(fig)

                # Plot Temperature vs. Power Consumption_Zone2
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(data=filtered_df, x='Temperature', y='PowerConsumption_Zone3', ax=ax)
                plt.title('Temperature vs. Power Consumption (Zone 3)')
                st.pyplot(fig)
                # Close the Matplotlib figure to release memory
                plt.close(fig)

                # Plot Humidity vs. Power Consumption_Zone2
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(data=filtered_df, x='Humidity', y='PowerConsumption_Zone2', ax=ax)
                plt.title('Humidity vs. Power Consumption (Zone 2)')
                st.pyplot(fig)
                # Close the Matplotlib figure to release memory
                plt.close(fig)

                # Plot Humidity vs. Power Consumption_Zone3
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(data=filtered_df, x='Humidity', y='PowerConsumption_Zone3', ax=ax)
                plt.title('Humidity vs. Power Consumption (Zone 3)')
                st.pyplot(fig)
                # Close the Matplotlib figure to release memory
                plt.close(fig)

                st.line_chart(df)

                # Plotting based on the filtered data
            if not filtered_df.empty:
                # Plot Temperature vs. Power Consumption_Zone1
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(data=filtered_df, x='Temperature', y='PowerConsumption_Zone1', ax=ax)
                plt.title('Temperature vs. Power Consumption (Zone 1)')
                st.pyplot(fig)
                # Close the Matplotlib figure to release memory
                plt.close(fig)

                # Plot Humidity vs. Power Consumption_Zone1
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(data=filtered_df, x='Humidity', y='PowerConsumption_Zone1', ax=ax)
                plt.title('Humidity vs. Power Consumption (Zone 1)')
                st.pyplot(fig)
                # Close the Matplotlib figure to release memory
                plt.close(fig)

            else:
                st.write("No data available for the selected month.")

            with col[0]:
                with elements("style_mui_sx1"):
                    mui.Box(
                        "Simple moving average (SMA) is a fundamental piece of information to smoothen predictions. It calculates the mean of all the data points in a time series in order to extract valuable information for the prediction algorithm.",
                        sx={
                            "bgcolor": "background.paper",
                            "boxShadow": 1,
                            "borderRadius": 2,
                            "p": 2,
                            "minWidth": 300,
                            "width": 100,

                        }

                    )
                with col[1]:
                    with elements("style_mui_sx2"):
                        mui.Box(
                            "",
                            sx={
                                "bgcolor": "background.paper",
                                "boxShadow": 1,
                                "borderRadius": 2,
                                "p": 2,
                                "minWidth": 300,
                                "width": 100

                            }
                        )

            with col[0]:
                create_card(
                    "https://electricityconsumptionanalysisdata.s3.eu-north-1.amazonaws.com/train-test-split.png",
                    "Train Test Split",
                    """The dataset is split into train and test sets.<b>NB: </b>70% of the data available is from January 1st until October 10th so that’s going to be our training set whereas from October 2nd until December 30th (the last data point available) will be our testing set. Zone 1 Power Consumption was selected as the target variable because proceeding area by area makes more sense.""")
            with col[1]:
                create_card(
                    "https://electricityconsumptionanalysisdata.s3.eu-north-1.amazonaws.com/predictionsvsdata.png",
                    "Predictions",
                    "Gradient Boosting algorithm was used to train the model. Overally, predictions follow the typical downtrend of the cold season even though the model has some trouble identifying the peaks in November and December.")

            create_card(
                "https://electricityconsumptionanalysisdata.s3.eu-north-1.amazonaws.com/octoberdatavspredictiondata.png",
                "October Data and Predictions",
                "If we zoom into the first 15 days of October we can clearly see how close the predictions are to the test data for that month.")
            create_card(
                "https://electricityconsumptionanalysisdata.s3.eu-north-1.amazonaws.com/decemberdatavsprediction.png",
                "December Data and Predictions",
                "In December, peak predictions are far away from the actual records. This can be improved in future models by either tuning hyperparameters or improving feature engineering")

            st.subheader("XGBoost Accuracy Metrics")

            metrics = st.columns(2)
            with metrics[0]:
                with elements("style_mui_sx3"):
                    mui.Box(
                        """
                        MAE =========> 1354.63 KWs,
                        r2 =============> 0.9101,
                        MSE ======> 3746515.4004,
                        RMSE=========> 1935.5917,
        """,
                        sx={
                            "bgcolor": "background.paper",
                            "boxShadow": 1,
                            "borderRadius": 2,
                            "p": 2,
                            "minWidth": 300,
                            "width": 150

                        }
                    )
            with metrics[1]:

                with elements("style_mui_sx5"):
                    acclst = """
                        <ul>
                            <li>Mean Absolute Error       - MAE</li>
                            <li>Mean Squared Error        - MSE</li>
                            <li>Percentage of variability - r2</li>                    
                            <li>Root Mean Squared Error   - RMSE</li>
                        </ul>
        
                        """
                    st.markdown(acclst, unsafe_allow_html=True)

            create_card("https://electricityconsumptionanalysisdata.s3.eu-north-1.amazonaws.com/predictedusage.png",
                        "2017 Predicted Usage", "")
            create_card(
                "https://electricityconsumptionanalysisdata.s3.eu-north-1.amazonaws.com/actualvspredictedusagedec.png",
                "2017 Predicted Usage Against Real Consumption", "")

            # download notebook for further analysis
            st.subheader("For Further Analysis, Download the Notebook bellow... happy debugging!!")

            def baloons():
                st.balloons()

            with open("profilereport.html", "rb") as file:
                st.download_button(label="Notebook",
                                   data=file,
                                   file_name="time-series-forecasting-on-power-consumption.ipynb",
                                   mime="Electricity_Forecasting_Notebook/notebook", on_click=baloons()
                                   )
        if menu == "Data":
            create_card(
                "https://electricityconsumptionanalysisdata.s3.eu-north-1.amazonaws.com/electricbackground1.jpg",
                "Electricity Consumption Dataset",
                "The data, which you can find at  <a href='https://www.kaggle.com/datasets/fedesoriano/electric-power-consumption'>this link</a>, has 52,416 energy consumption observations in 10-minute windows starting from January 1st, 2017, all the way through December 30th (not 31st) of the same year. Some of the features are:</p>    <ul>    <li>Date Time: Time window of ten minutes.</li>    <li>Temperature: Weather Temperature in °C</li>    <li>Humidity: Weather Humidity in %</li>    <li>Wind Speed: Wind Speed in km/h</li>    <li>Zone 1 Power Consumption in KiloWatts (KW)</li>    <li>Zone 2 Power Consumption in KW</li>    <li>Zone 3 Power Consumption in KW</li>    </ul>")

            df = load_data()
            profile = ProfileReport(df, title="Electricity consumption Profile Report")

            col1, col2 = st.columns(2)
            with col1:
                with open("profilereport.html", "rb") as file:
                    st.download_button(
                        label="Download Profile Report",
                        data=file,
                        file_name="profilereport.html",
                        mime="Electricity_profilereport/html"
                    )

            st.subheader("Data Card")
            st.write(filtered_df.head(20))

            csv = convert_df(df)
            with col2:
                st.download_button(
                    label="Download Dataset as CSV",
                    data=csv,
                    file_name="powerconsumption.csv",
                    mime='powerconsumption/csv',
                )

            if st.sidebar.checkbox("Summary Statistics"):
                st.subheader("Summary Card")
                st.write(df.describe())

        if menu == "About":
            st.title("About Page")
            st.markdown("---")
            about_page = """
            <div class="about">
                <h3>Introduction</h3>
                <p>The project's goal is to leverage time series analysis to predict energy consumption in 10-minute windows for the city of Tétouan in Morocco.</p>
                <h3>Context</h3>
                <p>
        According to a 2014 Census, Tétouan is a city in Morocco with a population of 380,000 people, occupying an area of 11,570 km². The city is located in the northern portion of the country and it faces the Mediterranean sea. The weather is particularly hot and humid during the summer.
        
        According to the state-owned ONEE, the “National Office of Electricity and Drinking Water”, Morocco’s energy production in 2019 came primarily from coal (38%), followed by hydroelectricity (16%), fuel oil (8 %), natural gas (18%), wind (11%), solar (7%), and others (2%) [1].
        
        Given the strong dependency on non-renewable sources (64%), forecasting energy consumption could help the stakeholders better manage purchases and stock. On top of that, Morocco’s plan is to reduce energy imports by increasing production from renewable sources. It’s common knowledge that sources like wind and solar present the risk of not being available all year round. Understanding the energy needs of the country, starting with a medium-sized city, could be a step further in planning these resources.
        </p>
            <h3>Data</h3>
            <p>The Supervisory Control and Data Acquisition System (SCADA) of Amendis, a public service operator, is responsible for collecting and providing the project’s power consumption data. The distribution network is powered by 3 zone stations, namely: Quads, Smir and Boussafou. The 3 zone stations power 3 different areas of the city, this is why we have three potential target variables.
        
        The data, which you can find at  <a href="https://www.kaggle.com/datasets/fedesoriano/electric-power-consumption">this link</a>, has 52,416 energy consumption observations in 10-minute windows starting from January 1st, 2017, all the way through December 30th (not 31st) of the same year. Some of the features are:</p>
        <ul>
        <li>Date Time: Time window of ten minutes.</li>
        <li>Temperature: Weather Temperature in °C</li>
        <li>Humidity: Weather Humidity in %</li>
        <li>Wind Speed: Wind Speed in km/h</li>
        <li>Zone 1 Power Consumption in KiloWatts (KW)</li>
        <li>Zone 2 Power Consumption in KW</li>
        <li>Zone 3 Power Consumption in KW</li>
        </ul>
            </div>
            """
            st.markdown(about_page, unsafe_allow_html=True)


        elif menu == "Contact us":
            st.title("Get in touch")
            contact_form = """
            <form action="https://formsubmit.co/faochieng@kabarak.ac.ke" method="POST">
                <input type="hidden" name="_captcha" value="false">
                <input type="text" name="name" placeholder="Your name" required>
                <input type="email" name="email" placeholder="Your email" required>
                <textarea name="message" placeholder="Your message here"></textarea>
                <button type="submit">Send</button>
            </form>
            """

            st.markdown(contact_form, unsafe_allow_html=True)

# Use Local CSS File
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
local_css("style.css")



if __name__ == "__main__":
    main()
