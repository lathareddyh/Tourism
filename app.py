# Import all the modules

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import warnings
import plotly.express as px

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Tourist Attraction", layout="wide")
st.title("🌍 Tourist Attraction System")

# Sidebar navigation
page = st.sidebar.radio(
    "Select a feature:",
    ["Home", "Rating Prediction", "Visit Mode Prediction", "Attraction Recommendations", "Analytics"]
)

# Load models and data
@st.cache_resource
def load_models():
    rating_model = joblib.load('XGB_model.pickle')
    visit_mode_model = joblib.load('xgb_clf_visit_mode_model.pickle')
    return rating_model, visit_mode_model

@st.cache_data
def load_tourist_data():
    # This assumes total_df is saved or accessible
    # For demo, you may need to load from CSV or database
    return pd.read_csv("cleaned_tourism_dataset.csv")


# Load the saved matrices
def load_recommendation_data():
    user_item_matrix = joblib.load('user_item_matrix.pkl')
    user_similarity_df = joblib.load('user_similarity_df.pkl')
    attraction_similarity_df = joblib.load('attraction_similarity_df.pkl')
    return user_item_matrix, user_similarity_df, attraction_similarity_df

# Recommendation Function based on user-based collaborative filtering
def recommend_attractions_user_based(user_id, top_n=5):
    # Similar users (excluding the user itself)
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:]
    
    # User's ratings
    user_ratings = user_item_matrix.loc[user_id]

    # Attractions not yet rated by user
    unrated_attractions = user_ratings[user_ratings.isna()].index

    scores = {}
    for attraction in unrated_attractions:
        score = 0
        for sim_user, similarity in similar_users.items():
            rating = user_item_matrix.loc[sim_user, attraction]
            
            if not pd.isna(rating):
                score += similarity * rating
        scores[attraction] = score
    
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

# Content-based recommendation function
def recommend_attractions_content_based(attraction_name, top_n=5):
    """
    Recommend attractions similar to the given attraction based on features
    """
    if attraction_name not in attraction_similarity_df.index:
        return []
    
    # Get similarity scores for the given attraction
    similar_attractions = attraction_similarity_df[attraction_name].sort_values(ascending=False)
    
    # Exclude the attraction itself and return top N
    return similar_attractions[1:top_n+1].index.tolist()

# Hybrid Recommendation System - Combining Collaborative and Content-Based Filtering

def recommend_attractions_hybrid(user_id, attraction_name=None, top_n=5, alpha=0.5):
    """
    Hybrid recommendation system combining user-based collaborative filtering 
    and content-based filtering.
    
    Parameters:
    - user_id: The user ID for which to generate recommendations
    - attraction_name: Optional - if provided, uses content-based as anchor
    - top_n: Number of recommendations to return
    - alpha: Weight for collaborative filtering (1-alpha for content-based)
    """
    
    # Get collaborative filtering recommendations
    collab_recs = recommend_attractions_user_based(user_id, top_n=top_n*2)
    collab_dict = {attraction: score for attraction, score in collab_recs}
    
    # Get content-based recommendations
    user_ratings = user_item_matrix.loc[user_id]
    user_visited = user_ratings[user_ratings.notna()].index.tolist()
    
    content_scores = {}
    
    if user_visited:
        # Get content-based recommendations for each attraction visited by user
        for visited_attraction in user_visited:
            if visited_attraction in attraction_similarity_df.index:
                similar = attraction_similarity_df[visited_attraction].sort_values(ascending=False)
                for attraction, similarity in similar[1:top_n+1].items():
                    if attraction not in user_ratings or pd.isna(user_ratings[attraction]):
                        content_scores[attraction] = content_scores.get(attraction, 0) + similarity
    
    # Normalize scores
    if collab_dict:
        max_collab = max(collab_dict.values()) if collab_dict.values() else 1
        collab_dict = {k: v/max_collab for k, v in collab_dict.items()}
    
    if content_scores:
        max_content = max(content_scores.values())
        content_scores = {k: v/max_content for k, v in content_scores.items()}
    
    # Combine scores using weighted average
    hybrid_scores = {}
    all_attractions = set(list(collab_dict.keys()) + list(content_scores.keys()))
    
    for attraction in all_attractions:
        collab_score = collab_dict.get(attraction, 0)
        content_score = content_scores.get(attraction, 0)
        hybrid_scores[attraction] = (alpha * collab_score) + ((1 - alpha) * content_score)
    
    # Return top N recommendations
    return sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]




rating_model, visit_mode_model = load_models()
tourist_data = load_tourist_data()

# Get Top Attractions
total_attr = tourist_data['Attraction'].value_counts().index
top10_attr = total_attr[:10]


# Home Page
if page == "Home":
    st.markdown("""
    ## Welcome to Tourist Attraction System
    
    This system helps predict user satisfaction and recommend attractions using advanced ML models.
    
    ### Features:
    
    ✨ **Rating Prediction** - Predict attraction ratings based on user demographics and visit details
    
    👥 **Visit Mode Prediction** - Classify the type of visit (Business, Family, Couples, Friends, Solo)
    
    🎯 **Attraction Recommendations** - Get personalized attraction recommendations using:
    - Collaborative Filtering (User-Based)
    - Content-Based Filtering
    - Hybrid Recommendation System
    - Analytics Dashboard with insights on popular attractions, regions, and user segments
    
    ### How to use:
    1. Select a feature from the sidebar
    2. Fill in the required information
    3. Get instant predictions and recommendations
    """)

# Rating Prediction Page
elif page == "Rating Prediction":
    st.header("🌟 Predict Attraction Rating")
    
    col1, col2 = st.columns(2)
    
    with col1:
        continent = st.selectbox("Continent", sorted(tourist_data["Continent"].unique().tolist()))
        region = st.selectbox("Region", sorted(tourist_data[tourist_data["Continent"] == continent]["Region"].unique().tolist()))
        country = st.selectbox("Country", sorted(tourist_data[tourist_data["Region"] == region]["Country"].unique().tolist()))    
        city = st.selectbox("City", sorted(tourist_data[tourist_data["Country"] == country]["CityName"].unique().tolist()))
    
    with col2:
        visit_year = st.number_input("Visit Year", min_value=2013, max_value=2023, value=2022)
        visit_month = st.slider("Visit Month", 1, 12, 6)
        visit_mode = st.selectbox("Visit Mode", sorted(tourist_data["VisitMode"].unique().tolist()))
        attraction_type = st.selectbox("AttractionType", sorted(tourist_data["AttractionType"].unique().tolist()))
    
    attraction = st.selectbox("Attraction", sorted(tourist_data[tourist_data["AttractionType"] == attraction_type]["Attraction"].unique().tolist()))

    
    if st.button("🔮 Predict Rating", key="rating_pred"):
        try:
            # Load encoders and scaler
            model_data = rating_model
            xgb_model = model_data['model']
            scaler = model_data['scaler']
            encoders = model_data['label_encoders']
            
            # Prepare sample data
            sample_data = pd.DataFrame({
                'Continent': [continent],
                'Region': [region],
                'Country': [country],
                'CityName': [city],
                'VisitYear': [visit_year],
                'VisitMonth': [visit_month],
                'VisitMode': [visit_mode],
                'AttractionType': [attraction_type],
                'Attraction': [attraction]
            })
            
            # Encode
            sample_encoded = sample_data.copy()
            for col in encoders.keys():
                try:
                    sample_encoded[col] = encoders[col].transform(sample_data[col])
                except:
                    st.warning(f"⚠️ '{col}' value not seen during training. Using default encoding.")
                    sample_encoded[col] = 0
            
            # Scale
            sample_scaled = scaler.transform(sample_encoded)
            sample_scaled = pd.DataFrame(sample_scaled, columns=scaler.get_feature_names_out())
            
            # Predict
            prediction = xgb_model.predict(sample_scaled)[0]
            
            st.success(f"### Predicted Rating: ⭐ {prediction:.2f}/5.0")
            
            # Rating interpretation
            if prediction >= 4.5:
                st.info("🎉 Excellent - Highly recommended!")
            elif prediction >= 4.0:
                st.info("😊 Very Good - Worth visiting")
            elif prediction >= 3.0:
                st.info("👍 Good - Nice to visit")
            else:
                st.warning("⚠️ Average - May want to explore alternatives")
                
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")

# Visit Mode Prediction Page
elif page == "Visit Mode Prediction":
    st.header("👥 Predict Visit Mode")
    
    col1, col2 = st.columns(2)
    
    with col1:
        continent_vm = st.selectbox("Continent", sorted(tourist_data["Continent"].unique().tolist()), key="vm_cont")
        region_vm = st.selectbox("Region", sorted(tourist_data[tourist_data["Continent"] == continent_vm]["Region"].unique().tolist()), key="vm_reg")
        country_vm = st.selectbox("Country", sorted(tourist_data[tourist_data["Region"] == region_vm]["Country"].unique().tolist()), key="vm_country")
        city_vm = st.selectbox("City Name", sorted(tourist_data[tourist_data["Country"] == country_vm]["CityName"].unique().tolist()), key="vm_city")
    

    with col2:
        visit_year_vm = st.number_input("Visit Year", min_value=2013, max_value=2023, value=2022, key="vm_year")
        visit_month_vm = st.slider("Visit Month", 1, 12, 6, key="vm_month")

        attraction_type_vm = st.selectbox("Attraction Type", sorted(tourist_data["AttractionType"].unique().tolist()), key="vm_atype")
        attraction_vm = st.selectbox("Attraction Name", sorted(tourist_data[tourist_data["AttractionType"] == attraction_type_vm]["Attraction"].unique().tolist()), key="vm_attr")

    
    if st.button("🎯 Predict Visit Mode", key="vm_pred"):
        try:
            # Load model components
            clf_data = visit_mode_model
            clf_model = clf_data['model']
            le_vm = clf_data['label_encoder']
            feat_encoders = clf_data['feature_encoders']
            
            # Prepare data
            sample_clf = pd.DataFrame({
                'Continent': [continent_vm],
                'Region': [region_vm],
                'Country': [country_vm],
                'CityName': [city_vm],
                'VisitYear': [visit_year_vm],
                'VisitMonth': [visit_month_vm],
                'AttractionType': [attraction_type_vm],
                'Attraction': [attraction_vm]
            })
            
            # Encode features
            for col, enc in feat_encoders.items():
                try:
                    sample_clf[col] = enc.transform(sample_clf[col])
                except:
                    st.warning(f"⚠️ '{col}' value not seen during training.")
                    sample_clf[col] = 0
            
            # Predict
            pred_enc = clf_model.predict(sample_clf)
            pred_label = le_vm.inverse_transform(pred_enc)[0]
            
            # Get prediction probabilities
            pred_proba = clf_model.predict_proba(sample_clf)[0]
            
            st.success(f"### Predicted Visit Mode: **{pred_label}**")
            
            # Show probabilities
            st.subheader("Confidence Scores:")
            proba_df = pd.DataFrame({
                'Visit Mode': le_vm.classes_,
                'Probability': pred_proba
            }).sort_values('Probability', ascending=False)
            
            st.bar_chart(proba_df.set_index('Visit Mode'))
            
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")

# Recommendations Page
elif page == "Attraction Recommendations":
    st.header("🎯 Attraction Recommendations")
    
    rec_type = st.radio("Recommendation Type:", 
        ["User-Based Collaborative Filtering", 
         "Content-Based Filtering", 
         "Hybrid Recommendations"])
    user_item_matrix, user_similarity_df, attraction_similarity_df = load_recommendation_data()


    if rec_type == "User-Based Collaborative Filtering":
        st.subheader("Based on Similar Users' Preferences")
        user_id = st.number_input("Enter User ID:", min_value=1, value=16)
        top_n = st.slider("Number of recommendations:", 1, 10, 5)



        if st.button("Get Recommendations"):
            # st.info("""
            # **Note:** To use this feature, you need to provide:
            # - A pre-built user-item rating matrix
            # - User similarity matrix (cosine similarity)
            
            # This requires loading the user_item_matrix and user_similarity_df from your session.
            # """)
            st.write(f"Would recommend {top_n} attractions for User {user_id}")

            # user_item_matrix, user_similarity_df, attraction_similarity_df = load_recommendation_data()

            recommendations = recommend_attractions_user_based(user_id=user_id, top_n=top_n)
            st.write("Recommended Attractions:")
            for i, (attraction, score) in enumerate(recommendations, 1):
                st.write(f"{i}. {attraction} (Score: {score:.2f})")












    
    elif rec_type == "Content-Based Filtering":
        st.subheader("Based on Attraction Similarity")
        # test_attraction = st.text_input("Enter attraction name:", "Sacred Monkey Forest Sanctuary")
        # content_attractions = tourist_data["Attraction"].unique().tolist()
        
        test_attraction =  st.selectbox("Attraction Name", sorted(tourist_data["Attraction"].unique().tolist()), key="con_attr")



        top_n_content = st.slider("Number of recommendations:", 1, 10, 5, key="content_n")
        




        if st.button("Get Similar Attractions"):
            # st.info("""
            # **Note:** To use this feature, you need to:
            # 1. Build an attraction feature matrix (AttractionType + Location)
            # 2. Compute attraction similarity using TF-IDF vectors
            
            # This requires the attraction_similarity_df from your session.
            # """)
            st.write(f"Finding {top_n_content} attractions similar to '{test_attraction}'")

            recommendations = recommend_attractions_content_based(test_attraction, top_n=top_n_content)
            st.write(f"Attractions similar to '{test_attraction}':")
            for i, attraction in enumerate(recommendations, 1):
                st.write(f"{i}. {attraction}")


    
    else:  # Hybrid
        st.subheader("Hybrid: Collaborative + Content-Based")
        hybrid_user_id = st.number_input("Enter User ID:", min_value=1, value=16, key="hybrid_user")

        # user_id_input = st.text_input("UserId (optional, for personalized hybrid recs)", value="")

        hybrid_alpha = st.slider("Weight for Collaborative Filtering (0=Content, 1=Collaborative):", 
                                0.0, 1.0, 0.6)
        top_n_hybrid = st.slider("Number of recommendations:", 1, 10, 5, key="hybrid_n")

        try:
            attractions = sorted(tourist_data["Attraction"].unique().tolist())
            attraction = st.selectbox("Attraction (optional)", [""] + attractions)
        except Exception:
            attraction = st.text_input("Attraction", value="")

        
        if st.button("Get Hybrid Recommendations"):
            # st.info("""
            # **Note:** Hybrid system combines:
            # - **Collaborative Filtering** (weight: {:.1f})
            # - **Content-Based Filtering** (weight: {:.1f})
            
            # To use this feature, you need both matrices from your notebook session.
            # """.format(hybrid_alpha, 1-hybrid_alpha))
            st.write(f"Generating {top_n_hybrid} recommendations for User {hybrid_user_id}")



            uid = int(hybrid_user_id)
            try:
                recs = recommend_attractions_hybrid(uid, attraction_name=None, top_n=top_n_hybrid, alpha=hybrid_alpha)
                # recs is list of tuples (attraction, score)
                recs = [r[0] for r in recs]
            except Exception:
                recs = []

            # If no user or hybrid failed, try content-based using provided attraction
            if not recs:
                try:
                    if attraction and attraction in attraction_similarity_df.index:
                        recs = recommend_attractions_content_based(attraction, top_n=5)
                    else:
                        # fallback to top popular attractions
                        recs = list(top10_attr[:5])
                except Exception:
                    recs = list(top10_attr[:5]) if 'top10_attr' in globals() else []

            st.subheader("Recommended Attractions")
            for i, r in enumerate(recs, 1):
                st.write(f"{i}. {r}")

# Analytics and Visualizations Page
elif page == "Analytics":
    st.header("📊 Tourism Analytics Dashboard")
    
    tab1, tab2, tab3 = st.tabs(["Popular Attractions", "Top Regions", "User Segments"])
    
    with tab1:
        st.subheader("Top Attractions by Visit Count")
        top_attractions = tourist_data['Attraction'].value_counts().head(10).sort_values(ascending=False)
        # Create Bar Chart
        fig = px.bar(top_attractions.reset_index(), x='Attraction', y='count', title="Top 10 Attractions by Visit Count", labels={'count': 'Visit Count'})
        st.plotly_chart(fig)

        st.subheader("Average Rating by Attraction")
        if 'Rating' in tourist_data.columns:
            avg_rating = tourist_data.groupby('Attraction')['Rating'].mean().round(2).sort_values(ascending=False).head(10)
            # Create Bar Chart
            fig = px.bar(avg_rating.reset_index(), x='Attraction', y='Rating', title="Average Rating by Attraction")
            st.plotly_chart(fig)
    with tab2:
        st.subheader("Visits by Region")
        region_visits = tourist_data['Region'].value_counts().head(10) 
        # Create Bar Chart
        fig = px.bar(region_visits.reset_index(), x='Region', y='count', title="Visits by Region", labels={'count': 'Visit Count'})
        st.plotly_chart(fig)
        
        st.subheader("Visits by Country")
        country_visits = tourist_data['Country'].value_counts().head(10)
        # Create Bar Chart
        fig = px.bar(country_visits.reset_index(), x='Country', y='count', title="Visits by Country", labels={'count': 'Visit Count'})
        st.plotly_chart(fig)
        
        st.subheader("Visits by Continent")
        continent_visits = tourist_data['Continent'].value_counts()
        # Create Pie Chart
        fig = px.pie(continent_visits, names=continent_visits.index, values=continent_visits.values, title="Visit Distribution by Continent")
        # Display
        st.plotly_chart(fig)
    
    with tab3:
        st.subheader("Visit Mode Distribution")
        visit_mode_dist = tourist_data['VisitMode'].value_counts()


        # Create Pie Chart
        fig = px.pie(visit_mode_dist, names=visit_mode_dist.index, values=visit_mode_dist.values, title="Visit Mode Distribution")
        # Display
        st.plotly_chart(fig)

        
        st.subheader("Attraction Type Distribution")
        attr_type_dist = tourist_data['AttractionType'].value_counts()
        # Create Bar Chart
        fig = px.bar(attr_type_dist.reset_index(), x='AttractionType', y='count', title="Attraction Type Distribution", labels={'count': 'Count'})
        st.plotly_chart(fig)
        
        st.subheader("Visits by Month")
        if 'VisitMonth' in tourist_data.columns:
            monthly_visits = tourist_data['VisitMonth'].value_counts().sort_index()
            fig = px.bar(monthly_visits.reset_index(), x='VisitMonth', y='count', title="Visits by Month", labels={'count': 'Visit Count'})
            st.plotly_chart(fig)


st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    """
    This application uses machine learning models trained on tourist attraction data.
    
    **Models Used:**
    - XGBoost Regressor (Rating Prediction)
    - XGBoost Classifier (Visit Mode Prediction)
    - Collaborative & Content-Based Filtering (Recommendations)
    """
)

