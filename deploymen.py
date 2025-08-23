import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from streamlit_option_menu import option_menu
import streamlit_lottie as st_lottie
import requests
from datetime import datetime

st.set_page_config(
    page_title='Movie Revenue Predictor',
    page_icon='🎬',
    initial_sidebar_state='collapsed'
)

def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

@st.cache_resource
def load_all():
    try:
        model = joblib.load("ValueClassifier")  
        encoder = joblib.load("encode")
        scaler = joblib.load("scaler")
        outlier_bounds = joblib.load("outlier_bounds")  
        return model, encoder, scaler, outlier_bounds
    except FileNotFoundError:
        st.error("Model files not found. Please ensure the following files exist:")
        st.error("- ValueClassifier")
        st.error("- encode") 
        st.error("- scaler")
        st.error("- outlier_bounds")
        return None, None, None, None

model, encoder, scaler, outlier_bounds = load_all()


def preprocess_input_data(company_name, budget, original_language, popularity,
                         runtime, vote_average, vote_count, country,
                         belongs_to_collection, country_of_company, release_year,
                         genre_count):
    """
    Preprocess input data according to the notebook structure
    """
    features = pd.DataFrame([{
        'company_name': company_name,
        'budget': budget,
        'original_language': original_language,
        'popularity': popularity,
        'runtime': runtime,
        'vote_average': vote_average,
        'vote_count': vote_count,
        'country': country,
        'belongs_to_collection': belongs_to_collection,
        'country_of_company': country_of_company,
        'release_year': release_year,
        'genre_count': genre_count
    }])
    
    features['belongs_to_collection'] = features['belongs_to_collection'].map({'Yes': 1, 'No': 0})
    

    if outlier_bounds:
        for col in ['budget', 'popularity', 'runtime', 'vote_average', 'vote_count']:
            if col in outlier_bounds:
                lower_bound, upper_bound = outlier_bounds[col]
                features[col] = features[col].apply(
                    lambda x: lower_bound if x < lower_bound else (upper_bound if x > upper_bound else x)
                )
    
    return features

def predict_revenue(company_name, budget, original_language, popularity,
                    runtime, vote_average, vote_count, country,
                    belongs_to_collection, country_of_company, release_year,
                    genre_count):
    """
    Predict revenue using the trained model, including all columns used in training.
    """
    if model is None or encoder is None or scaler is None:
        st.error("Model not loaded. Cannot make predictions.")
        return None
    
    try:
        features = pd.DataFrame([{
            'company_name': company_name,
            'budget': budget,
            'original_language': original_language,
            'popularity': popularity,
            'runtime': runtime,
            'vote_average': vote_average,
            'vote_count': vote_count,
            'country': country,
            'belongs_to_collection': belongs_to_collection,
            'country_of_company': country_of_company,
            'release_year': release_year,
            'genre_count': genre_count
        }])

        categorical_cols = ['company_name', 'original_language', 'country', 'country_of_company']
        features_encoded = encoder.transform(features[categorical_cols])
        encoded_cols = encoder.get_feature_names_out(categorical_cols)
        features_encoded_df = pd.DataFrame(features_encoded, columns=encoded_cols)

        features['belongs_to_collection'] = features['belongs_to_collection'].map({'Yes': 1, 'No': 0})

        numeric_cols = ['budget', 'vote_average', 'vote_count', 'popularity', 'runtime', 'release_year', 'genre_count']
        features_numeric = features[numeric_cols].reset_index(drop=True)

        features_combined = pd.concat([features_encoded_df, features_numeric, features[['belongs_to_collection']].reset_index(drop=True)], axis=1)

        scale_cols = ['company_name', 'country', 'budget', 'vote_average', 'vote_count',
                      'country_of_company', 'original_language', 'popularity','release_year' , 'genre_count']
        features_for_scaling = features_combined[scale_cols]
        features_scaled = scaler.transform(features_for_scaling)
        scaled_df = pd.DataFrame(features_scaled, columns=scale_cols)

        remaining_cols = features_combined.drop(columns=scale_cols).reset_index(drop=True)
        features_final = pd.concat([scaled_df, remaining_cols], axis=1)

        for col in model.feature_names_in_:
            if col not in features_final.columns:
                features_final[col] = 0

        features_final = features_final[model.feature_names_in_]

        features_final = features_final.astype(float)

        prediction = model.predict(features_final)
        return prediction[0]

    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None



with st.sidebar:
    choose = option_menu(
        None,
        ["Home", "Graphs", "Model Info"],
        icons=['house', 'graph-up', 'info-circle'],
        menu_icon="app-indicator",
        default_index=0,
        styles={
            "container": {"padding": "5!important", "background-color": "#fafafa"},
            "icon": {"color": "orange", "font-size": "25px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#FF4B4B", "color": "white"}
        }
    )


if choose == 'Home':
    st.title('🎬 Movie Revenue Predictor')
    st.markdown("---")
    st.subheader('Enter movie details to predict Revenue')
    
    col1, col2 = st.columns(2)
    
    with col1:
        company_name = st.text_input('Company Name', placeholder='e.g., Warner Bros.')
        budget = st.number_input('Budget ($)', min_value=0.0, value=50000000.0, step=1000000.0)
        original_language = st.selectbox('Original Language', 
                                       ['en', 'fr', 'de', 'ja', 'zh', 'it', 'pt', 'ko', 'cn', 'es', 'th', 'hi', 'ru', 'sv'])
        popularity = st.number_input('Popularity Score', min_value=0.0, value=50.0, step=1.0)
        runtime = st.number_input('Runtime (minutes)', min_value=0, value=120, step=5)
        vote_average = st.number_input('Vote Average (0-10)', min_value=0.0, max_value=10.0, value=6.5, step=0.1)
    
    with col2:
        vote_count = st.number_input('Vote Count', min_value=0, value=1000, step=100)
        country = st.selectbox('Country', ['US', 'FR', 'DE', 'GB', 'JP', 'CA', 'AU', 'KR', 'MX', 'CN', 'IN', 'BR'])
        belongs_to_collection = st.selectbox('Belongs to Collection', ['No', 'Yes'])
        country_of_company = st.selectbox('Country of Company', ['US', 'FR', 'DE', 'GB', 'JP', 'CA', 'AU', 'KR', 'MX', 'CN', 'IN', 'BR'])
        release_year = st.number_input('Release Year', min_value=1900, max_value=2030, value=2024)
        genre_count = st.number_input('Number of Genres', min_value=1, max_value=12, value=2)
    
    st.markdown("---")
    
    if st.button('🎯 Predict Revenue', type='primary', use_container_width=True):
        if model is not None:
            with st.spinner('Calculating prediction...'):
                result = predict_revenue(
                    company_name, budget, original_language, popularity,
                    runtime, vote_average, vote_count, country,
                    belongs_to_collection, country_of_company, release_year,
                    genre_count
                )
            
            if result is not None:

                if result > 1e9:
                    formatted_result = f"${result/1e9:.2f}B"
                    st.success(f'🎉 **Predicted Revenue: {formatted_result}**')
                elif result > 1e6:
                    formatted_result = f"${result/1e6:.2f}M"
                    st.success(f'🎉 **Predicted Revenue: {formatted_result}**')
                else:
                    formatted_result = f"${result:,.0f}"
                    st.success(f'🎉 **Predicted Revenue: {formatted_result}**')
                
        else:
            st.error("Model not available. Please check if all model files are present.")

elif choose == "Graphs":
    st.title('📊Visualizations')
    st.markdown("---")
    
    st.subheader("Revenue Analysis")
    
    try:
        @st.cache_data
        def load_movie_data():
            movie_details = pd.read_csv('movie_details.csv')
            companies = pd.read_csv('companies.csv')
            countries = pd.read_csv('countries.csv')
            genres = pd.read_csv('genres.csv')
            langs = pd.read_csv('langs.csv')
            
            for table in [companies, countries, genres, langs]:
                movie_details = pd.merge(
                    movie_details, table,
                    left_on="id", right_on="movie_id", how="left"
                )
                movie_details.drop(columns=["movie_id"], inplace=True)
            
            return movie_details
        
        movie_data = load_movie_data()
        
        st.subheader("Generated Visualizations")
        
        viz_files = [
            "Average_Revenue_Collection_vs_Non-Collection.png",
            "Average_Revenue_per_Country_(Top 15).png",
            "Budget_vs_Revenue.png",
            "Number_of_Movies_per_Country.png",
            "boxplot_budget.png",
            "boxplot_popularity.png",
            "boxplot_revenue.png",
            "boxplot_runtime.png",
            "boxplot_vote_average.png",
            "boxplot_vote_count.png",
            "Number_of_Movies_per_Year.png",
            "Popularity_vs_Revenue.png",
            "Top_10_Companies_by_Movie_Count.png",
            "Total_Revenue_per_Month.png",
            "Total_Revenue_per_Year.png",
            "heatmap.png",
            "Pie_chart_collection.png",
            "Histogram_budget.png",
            "Histogram_popularity.png",
            "Histogram_revenue.png",
            "Histogram_runtime.png",
            "Histogram_vote_average.png",
            "Histogram_vote_count.png" ,
            "compare_between_models.png"
        ]
        
        for viz_file in viz_files:
            try:
                st.image(viz_file, caption=viz_file.replace('.png', '').replace('_', ' '))
            except:
                st.warning(f"Visualization file '{viz_file}' not found")
                
    except FileNotFoundError:
        st.error("Data files not found. Please ensure all CSV files are present in the current directory.")

elif choose == "Model Info":
    st.title('🤖 Model Information')
    st.markdown("---")
    
    st.subheader("Model Details")
    
    if model is not None:
        st.success("✅ Model loaded successfully!")
        
        st.info(f"**Model Type**: {type(model).__name__}")
        
        if hasattr(model, 'n_estimators'):
            st.info(f"**Number of Estimators**: {model.n_estimators}")
        if hasattr(model, 'max_depth'):
            st.info(f"**Max Depth**: {model.max_depth}")
        
        st.info("**Features Used (12 total)**:")
        feature_list = [
            'company_name', 'budget', 'original_language', 'popularity',
       'runtime', 'vote_average', 'vote_count', 'country',
       'belongs_to_collection', 'country_of_company', 'release_year',
       'genre_count'
        ]
        for feature in feature_list:
            st.write(f"• {feature}")
        
        st.info("**Preprocessing**:")
        st.write("• Categorical encoding using TargetEncoder")
        st.write("• Feature scaling using StandardScaler")
        st.write("• Outlier handling using IQR method")
        
    else:
        st.error("❌ Model not loaded")
    
    st.markdown("---")
    st.subheader("Model Performance")
    st.write("Based on the notebook analysis:")
    
    col1 = st.columns(1)[0]
    with col1:
        st.metric("Random Forest (Train)", "0.8575")
        st.metric("Random Forest (Test)", "0.8169")
        st.metric("Cross-Validation Accuracy (5-Fold)", "0.8105")
        st.text("Cross-Validation Scores (5-Fold): [0.7901, 0.8198, 0.8367, 0.7603, 0.8455]")
        st.metric("Train RMSE:" ," 60491384.3659")
        st.metric("Test  RMSE:" ," 67053723.1282")
        st.metric("Train RMSE (of mean revenue) :"," 40.43%")
        st.metric("Test RMSE (of mean revenue) :"," 44.81%")
    
    st.info("The model shows good performance with reasonable generalization from training to test data.")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🎬 Movie Revenue Predictor | Built with Streamlit | Based on Movie Dataset Analysis</p>
</div>
""", unsafe_allow_html=True)
