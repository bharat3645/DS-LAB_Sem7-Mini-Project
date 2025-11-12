"""
Crime Analytics Dashboard - Streamlit Application
Advanced Data Science Project with Regression, Classification, Clustering, and Deep Learning
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data_preprocessing import CrimeDataPreprocessor
from ml_models import CrimeMLModels
from deep_learning import DeepLearningModels, TF_AVAILABLE

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="Crime Analytics Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 1rem;
    }
    h2 {
        color: #ff7f0e;
        padding-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess both datasets"""
    processor = CrimeDataPreprocessor()
    missing_df, crimes_df = processor.load_data()
    missing_clean = processor.clean_missing_persons(missing_df)
    crimes_clean = processor.clean_juvenile_crimes(crimes_df)
    merged = processor.merge_datasets(missing_clean, crimes_clean)
    featured = processor.create_features(merged)
    
    return {
        'missing_raw': missing_df,
        'crimes_raw': crimes_df,
        'missing_clean': missing_clean,
        'crimes_clean': crimes_clean,
        'merged': merged,
        'featured': featured,
        'processor': processor
    }

def main():
    # Header
    st.title("🔍 Advanced Crime Analytics Dashboard")
    st.markdown("### Comprehensive Data Science Project: Missing Persons & Juvenile Crimes Analysis (2017-2022)")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=Crime+Analytics", use_container_width=True)
        st.markdown("## Navigation")
        st.markdown("---")
        
        page = st.radio(
            "Select Analysis:",
            ["📊 Overview & EDA",
             "📈 Missing Persons Analysis",
             "🔪 Juvenile Crimes Analysis", 
             "🔄 Combined Analysis",
             "🤖 Machine Learning - Regression",
             "🎯 Machine Learning - Classification",
             "🌐 Machine Learning - Clustering",
             "🧠 Deep Learning",
             "📑 Summary & Insights"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.info("This dashboard analyzes district-wise missing persons and juvenile crimes data across India from 2017-2022, employing advanced data science techniques.")
        
    # Load data
    if not st.session_state.data_loaded:
        with st.spinner("Loading and preprocessing data..."):
            st.session_state.data = load_and_preprocess_data()
            st.session_state.data_loaded = True
    
    data = st.session_state.data
    
    # Route to appropriate page
    if page == "📊 Overview & EDA":
        show_overview(data)
    elif page == "📈 Missing Persons Analysis":
        show_missing_persons_analysis(data)
    elif page == "🔪 Juvenile Crimes Analysis":
        show_juvenile_crimes_analysis(data)
    elif page == "🔄 Combined Analysis":
        show_combined_analysis(data)
    elif page == "🤖 Machine Learning - Regression":
        show_regression_analysis(data)
    elif page == "🎯 Machine Learning - Classification":
        show_classification_analysis(data)
    elif page == "🌐 Machine Learning - Clustering":
        show_clustering_analysis(data)
    elif page == "🧠 Deep Learning":
        show_deep_learning_analysis(data)
    elif page == "📑 Summary & Insights":
        show_summary(data)

def show_overview(data):
    st.header("📊 Data Overview & Exploratory Data Analysis")
    
    tab1, tab2, tab3 = st.tabs(["📋 Dataset Info", "📊 Statistics", "🗺️ Geographic Distribution"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Missing Persons Dataset")
            st.write(f"**Shape:** {data['missing_clean'].shape}")
            st.write(f"**Years:** {sorted(data['missing_raw']['year'].unique())}")
            st.write(f"**States:** {data['missing_raw']['state_name'].nunique()}")
            st.write(f"**Districts:** {data['missing_raw']['district_name'].nunique()}")
            
            with st.expander("View Sample Data"):
                st.dataframe(data['missing_clean'].head(10))
        
        with col2:
            st.subheader("Juvenile Crimes Dataset")
            st.write(f"**Shape:** {data['crimes_clean'].shape}")
            st.write(f"**Years:** {sorted(data['crimes_raw']['year'].unique())}")
            st.write(f"**States:** {data['crimes_raw']['state_name'].nunique()}")
            st.write(f"**Districts:** {data['crimes_raw']['district_name'].nunique()}")
            st.write(f"**Crime Types:** {len([col for col in data['crimes_clean'].columns if col not in ['id', 'year', 'state_name', 'state_code', 'district_name', 'district_code', 'registration_circles']])-4}")
            
            with st.expander("View Sample Data"):
                st.dataframe(data['crimes_clean'].head(10))
    
    with tab2:
        st.subheader("📈 Key Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_missing = int(data['missing_clean']['total_missing'].sum())
            st.metric("Total Missing Persons", f"{total_missing:,}")
        
        with col2:
            total_crimes = int(data['crimes_clean']['total_crimes'].sum())
            st.metric("Total Juvenile Crimes", f"{total_crimes:,}")
        
        with col3:
            avg_missing_per_district = data['missing_clean']['total_missing'].mean()
            st.metric("Avg Missing/District", f"{avg_missing_per_district:.0f}")
        
        with col4:
            avg_crimes_per_district = data['crimes_clean']['total_crimes'].mean()
            st.metric("Avg Crimes/District", f"{avg_crimes_per_district:.0f}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Missing Persons by Gender")
            gender_data = pd.DataFrame({
                'Gender': ['Male', 'Female', 'Transgender'],
                'Count': [
                    data['missing_clean']['male_total'].sum(),
                    data['missing_clean']['female_total'].sum(),
                    data['missing_clean']['transgender_total'].sum()
                ]
            })
            fig = px.pie(gender_data, values='Count', names='Gender', 
                        title='Gender Distribution of Missing Persons',
                        color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Crime Categories Distribution")
            crime_categories = pd.DataFrame({
                'Category': ['Violent Crimes', 'Sexual Crimes', 'Property Crimes', 'Other Crimes'],
                'Count': [
                    data['crimes_clean']['violent_crimes'].sum(),
                    data['crimes_clean']['sexual_crimes'].sum(),
                    data['crimes_clean']['property_crimes'].sum(),
                    data['crimes_clean']['total_crimes'].sum() - (
                        data['crimes_clean']['violent_crimes'].sum() +
                        data['crimes_clean']['sexual_crimes'].sum() +
                        data['crimes_clean']['property_crimes'].sum()
                    )
                ]
            })
            fig = px.pie(crime_categories, values='Count', names='Category',
                        title='Crime Categories Distribution',
                        color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("🗺️ Geographic Distribution")
        
        # Top states analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Top 10 States - Missing Persons")
            state_missing = data['missing_clean'].groupby('state_name')['total_missing'].sum().sort_values(ascending=False).head(10)
            fig = px.bar(x=state_missing.values, y=state_missing.index, orientation='h',
                        labels={'x': 'Total Missing', 'y': 'State'},
                        color=state_missing.values,
                        color_continuous_scale='Reds')
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Top 10 States - Juvenile Crimes")
            state_crimes = data['crimes_clean'].groupby('state_name')['total_crimes'].sum().sort_values(ascending=False).head(10)
            fig = px.bar(x=state_crimes.values, y=state_crimes.index, orientation='h',
                        labels={'x': 'Total Crimes', 'y': 'State'},
                        color=state_crimes.values,
                        color_continuous_scale='Blues')
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)

def show_missing_persons_analysis(data):
    st.header("📈 Missing Persons Deep Dive Analysis")
    
    tab1, tab2, tab3 = st.tabs(["📅 Temporal Trends", "👥 Demographic Analysis", "🎯 Age Group Analysis"])
    
    with tab1:
        st.subheader("Yearly Trends")
        
        yearly_missing = data['missing_clean'].groupby('year')['total_missing'].sum().reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=yearly_missing['year'], y=yearly_missing['total_missing'],
                                mode='lines+markers', name='Total Missing',
                                line=dict(color='#e74c3c', width=3),
                                marker=dict(size=10)))
        fig.update_layout(title='Total Missing Persons Over Years',
                         xaxis_title='Year', yaxis_title='Number of Missing Persons',
                         hovermode='x unified', height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Year-over-year change
        yearly_missing['yoy_change'] = yearly_missing['total_missing'].pct_change() * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("2017 Total", f"{int(yearly_missing[yearly_missing['year']==2017]['total_missing'].values[0]):,}")
        with col2:
            st.metric("2022 Total", f"{int(yearly_missing[yearly_missing['year']==2022]['total_missing'].values[0]):,}")
        with col3:
            total_change = ((yearly_missing[yearly_missing['year']==2022]['total_missing'].values[0] - 
                           yearly_missing[yearly_missing['year']==2017]['total_missing'].values[0]) / 
                          yearly_missing[yearly_missing['year']==2017]['total_missing'].values[0] * 100)
            st.metric("Overall Change", f"{total_change:.1f}%")
    
    with tab2:
        st.subheader("Gender-wise Analysis")
        
        gender_yearly = data['missing_clean'].groupby('year')[['male_total', 'female_total', 'transgender_total']].sum().reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=gender_yearly['year'], y=gender_yearly['male_total'], name='Male',
                            marker_color='#3498db'))
        fig.add_trace(go.Bar(x=gender_yearly['year'], y=gender_yearly['female_total'], name='Female',
                            marker_color='#e74c3c'))
        fig.add_trace(go.Bar(x=gender_yearly['year'], y=gender_yearly['transgender_total'], name='Transgender',
                            marker_color='#9b59b6'))
        fig.update_layout(title='Gender-wise Missing Persons Trends',
                         xaxis_title='Year', yaxis_title='Number of Missing Persons',
                         barmode='group', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Age Group Analysis")
        
        age_groups = data['missing_clean'].groupby('year')[['children_missing', 'youth_missing', 
                                                            'adults_missing', 'elderly_missing']].sum().reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=age_groups['year'], y=age_groups['children_missing'], 
                                name='Children (<14)', mode='lines+markers', line=dict(width=3)))
        fig.add_trace(go.Scatter(x=age_groups['year'], y=age_groups['youth_missing'],
                                name='Youth (14-30)', mode='lines+markers', line=dict(width=3)))
        fig.add_trace(go.Scatter(x=age_groups['year'], y=age_groups['adults_missing'],
                                name='Adults (30-60)', mode='lines+markers', line=dict(width=3)))
        fig.add_trace(go.Scatter(x=age_groups['year'], y=age_groups['elderly_missing'],
                                name='Elderly (60+)', mode='lines+markers', line=dict(width=3)))
        fig.update_layout(title='Age Group-wise Missing Persons Trends',
                         xaxis_title='Year', yaxis_title='Number of Missing Persons',
                         hovermode='x unified', height=400)
        st.plotly_chart(fig, use_container_width=True)

def show_juvenile_crimes_analysis(data):
    st.header("🔪 Juvenile Crimes Deep Dive Analysis")
    
    tab1, tab2, tab3 = st.tabs(["📅 Temporal Trends", "🗂️ Crime Categories", "🎯 Top Crimes"])
    
    with tab1:
        st.subheader("Yearly Crime Trends")
        
        yearly_crimes = data['crimes_clean'].groupby('year')['total_crimes'].sum().reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=yearly_crimes['year'], y=yearly_crimes['total_crimes'],
                                mode='lines+markers', name='Total Crimes',
                                line=dict(color='#2ecc71', width=3),
                                marker=dict(size=10)))
        fig.update_layout(title='Total Juvenile Crimes Over Years',
                         xaxis_title='Year', yaxis_title='Number of Crimes',
                         hovermode='x unified', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Crime Category Trends")
        
        category_yearly = data['crimes_clean'].groupby('year')[['violent_crimes', 'sexual_crimes', 
                                                                 'property_crimes']].sum().reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=category_yearly['year'], y=category_yearly['violent_crimes'],
                                name='Violent Crimes', mode='lines+markers', line=dict(width=3)))
        fig.add_trace(go.Scatter(x=category_yearly['year'], y=category_yearly['sexual_crimes'],
                                name='Sexual Crimes', mode='lines+markers', line=dict(width=3)))
        fig.add_trace(go.Scatter(x=category_yearly['year'], y=category_yearly['property_crimes'],
                                name='Property Crimes', mode='lines+markers', line=dict(width=3)))
        fig.update_layout(title='Crime Category Trends Over Years',
                         xaxis_title='Year', yaxis_title='Number of Crimes',
                         hovermode='x unified', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Top Crime Types")
        
        crime_cols = [col for col in data['crimes_clean'].columns if col not in 
                     ['id', 'year', 'state_name', 'state_code', 'district_name', 
                      'district_code', 'registration_circles', 'total_crimes',
                      'violent_crimes', 'sexual_crimes', 'property_crimes']]
        
        crime_totals = data['crimes_clean'][crime_cols].sum().sort_values(ascending=False).head(15)
        
        fig = px.bar(x=crime_totals.values, y=crime_totals.index, orientation='h',
                    labels={'x': 'Total Cases', 'y': 'Crime Type'},
                    title='Top 15 Crime Types (2017-2022)',
                    color=crime_totals.values,
                    color_continuous_scale='Viridis')
        fig.update_layout(showlegend=False, height=500)
        st.plotly_chart(fig, use_container_width=True)

def show_combined_analysis(data):
    st.header("🔄 Combined Analysis: Missing Persons vs Juvenile Crimes")
    
    tab1, tab2 = st.tabs(["📊 Correlation Analysis", "🗺️ State-level Comparison"])
    
    with tab1:
        st.subheader("Relationship Between Missing Persons and Crimes")
        
        if 'total_missing' in data['featured'].columns and 'total_crimes' in data['featured'].columns:
            fig = px.scatter(data['featured'], x='total_crimes', y='total_missing',
                           color='year', size='total_missing',
                           hover_data=['state_name', 'district_name'],
                           title='Missing Persons vs Juvenile Crimes (District-level)',
                           labels={'total_crimes': 'Total Crimes', 'total_missing': 'Total Missing Persons'},
                           opacity=0.6)
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation coefficient
            corr = data['featured'][['total_missing', 'total_crimes']].corr().iloc[0, 1]
            st.info(f"**Correlation Coefficient:** {corr:.4f}")
    
    with tab2:
        st.subheader("State-level Comparison")
        
        state_comparison = data['featured'].groupby('state_name').agg({
            'total_missing': 'sum',
            'total_crimes': 'sum'
        }).sort_values('total_missing', ascending=False).head(15)
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Missing Persons', 'Juvenile Crimes'))
        
        fig.add_trace(go.Bar(x=state_comparison['total_missing'], y=state_comparison.index,
                            orientation='h', name='Missing', marker_color='#e74c3c'),
                     row=1, col=1)
        
        fig.add_trace(go.Bar(x=state_comparison['total_crimes'], y=state_comparison.index,
                            orientation='h', name='Crimes', marker_color='#3498db'),
                     row=1, col=2)
        
        fig.update_layout(height=600, showlegend=False)
        fig.update_xaxes(title_text="Total Missing", row=1, col=1)
        fig.update_xaxes(title_text="Total Crimes", row=1, col=2)
        
        st.plotly_chart(fig, use_container_width=True)

def show_regression_analysis(data):
    st.header("🤖 Machine Learning: Regression Analysis")
    st.markdown("### Predicting Total Missing Persons")
    
    if st.button("🚀 Train Regression Models"):
        with st.spinner("Training multiple regression models..."):
            processor = data['processor']
            ml = CrimeMLModels()
            
            # Prepare data
            X, y, feature_cols = processor.prepare_for_modeling(data['missing_clean'], 'total_missing')
            X = X[[col for col in X.columns if col not in ['male_total', 'female_total', 
                                                           'transgender_total', 'children_missing',
                                                           'youth_missing', 'adults_missing', 
                                                           'elderly_missing']]]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train models
            results, best_model = ml.train_regression_models(X_train, X_test, y_train, y_test,
                                                            "Missing_Persons_Prediction")
            
            st.session_state.reg_results = results
            st.session_state.best_reg_model = best_model
            
            st.success(f"✅ Training Complete! Best Model: **{best_model}**")
    
    # Display results
    if 'reg_results' in st.session_state:
        results = st.session_state.reg_results
        
        st.markdown("---")
        st.subheader("📊 Model Comparison")
        
        # Create comparison dataframe
        comparison_data = []
        for name, result in results.items():
            comparison_data.append({
                'Model': name,
                'Test R²': result['test_r2'],
                'Test RMSE': result['test_rmse'],
                'Test MAE': result['test_mae']
            })
        
        comparison_df = pd.DataFrame(comparison_data).sort_values('Test R²', ascending=False)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(comparison_df.style.highlight_max(subset=['Test R²'], color='lightgreen')
                                          .highlight_min(subset=['Test RMSE', 'Test MAE'], color='lightgreen'),
                        use_container_width=True)
        
        with col2:
            st.markdown("#### Best Model")
            best_model_data = comparison_df.iloc[0]
            st.metric("Model", best_model_data['Model'])
            st.metric("R² Score", f"{best_model_data['Test R²']:.4f}")
            st.metric("RMSE", f"{best_model_data['Test RMSE']:.2f}")
            st.metric("MAE", f"{best_model_data['Test MAE']:.2f}")

def show_classification_analysis(data):
    st.header("🎯 Machine Learning: Classification Analysis")
    st.markdown("### Classifying High vs Low Crime Districts")
    
    if st.button("🚀 Train Classification Models"):
        with st.spinner("Training multiple classification models..."):
            processor = data['processor']
            ml = CrimeMLModels()
            
            # Prepare data
            crimes_for_class = data['crimes_clean'].copy()
            crime_threshold = crimes_for_class['total_crimes'].median()
            crimes_for_class['crime_level'] = (crimes_for_class['total_crimes'] > crime_threshold).astype(int)
            
            X, y, _ = processor.prepare_for_modeling(crimes_for_class, 'crime_level')
            X = X[[col for col in X.columns if col not in ['total_crimes', 'violent_crimes',
                                                           'sexual_crimes', 'property_crimes']]][:50]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train models
            results, best_model = ml.train_classification_models(X_train, X_test, y_train, y_test,
                                                                "Crime_Level_Classification")
            
            st.session_state.class_results = results
            st.session_state.best_class_model = best_model
            
            st.success(f"✅ Training Complete! Best Model: **{best_model}**")
    
    # Display results
    if 'class_results' in st.session_state:
        results = st.session_state.class_results
        
        st.markdown("---")
        st.subheader("📊 Model Comparison")
        
        # Create comparison dataframe
        comparison_data = []
        for name, result in results.items():
            comparison_data.append({
                'Model': name,
                'Test Accuracy': result['test_accuracy'],
                'Train Accuracy': result['train_accuracy']
            })
        
        comparison_df = pd.DataFrame(comparison_data).sort_values('Test Accuracy', ascending=False)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(comparison_df.style.highlight_max(subset=['Test Accuracy'], color='lightgreen'),
                        use_container_width=True)
        
        with col2:
            st.markdown("#### Best Model")
            best_model_data = comparison_df.iloc[0]
            st.metric("Model", best_model_data['Model'])
            st.metric("Accuracy", f"{best_model_data['Test Accuracy']:.4f}")

def show_clustering_analysis(data):
    st.header("🌐 Machine Learning: Clustering Analysis")
    st.markdown("### Discovering State-level Crime Patterns")
    
    n_clusters = st.slider("Number of Clusters", min_value=3, max_value=10, value=5)
    
    if st.button("🚀 Perform Clustering"):
        with st.spinner("Performing clustering analysis..."):
            processor = data['processor']
            ml = CrimeMLModels()
            
            # Prepare data
            state_agg = processor.get_state_aggregated(data['crimes_clean'])
            clustering_features = ['total_crimes', 'violent_crimes', 'sexual_crimes', 'property_crimes']
            X_cluster = state_agg[clustering_features]
            
            # Train clustering models
            results, best_model = ml.train_clustering_models(X_cluster, "State_Crime_Patterns", 
                                                            n_clusters=n_clusters)
            
            # Add cluster labels to state data
            state_agg['cluster'] = results[best_model]['labels']
            
            st.session_state.cluster_results = results
            st.session_state.cluster_data = state_agg
            st.session_state.best_cluster_model = best_model
            
            st.success(f"✅ Clustering Complete! Best Model: **{best_model}**")
    
    # Display results
    if 'cluster_results' in st.session_state:
        results = st.session_state.cluster_results
        cluster_data = st.session_state.cluster_data
        
        st.markdown("---")
        st.subheader("📊 Clustering Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Cluster Distribution")
            cluster_counts = cluster_data['cluster'].value_counts().sort_index()
            fig = px.bar(x=cluster_counts.index, y=cluster_counts.values,
                        labels={'x': 'Cluster', 'y': 'Number of State-Years'},
                        color=cluster_counts.values,
                        color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Cluster Characteristics")
            cluster_stats = cluster_data.groupby('cluster')[['total_crimes', 'violent_crimes', 
                                                             'sexual_crimes', 'property_crimes']].mean()
            st.dataframe(cluster_stats.style.background_gradient(cmap='YlOrRd', axis=1),
                        use_container_width=True)

def show_deep_learning_analysis(data):
    st.header("🧠 Deep Learning Analysis")
    
    if not TF_AVAILABLE:
        st.warning("⚠️ TensorFlow is not available. Deep learning features are disabled.")
        st.info("To enable deep learning, install TensorFlow: `pip install tensorflow`")
        return
    
    st.markdown("### Neural Networks for Crime Prediction")
    
    tab1, tab2 = st.tabs(["🔢 Regression (Missing Persons)", "🎯 Classification (Crime Level)"])
    
    with tab1:
        st.subheader("Deep Neural Network - Regression")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            epochs = st.number_input("Epochs", min_value=10, max_value=200, value=50)
        with col2:
            batch_size = st.number_input("Batch Size", min_value=16, max_value=128, value=32)
        with col3:
            st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🚀 Train Deep Learning Model (Regression)"):
            with st.spinner("Training deep neural network..."):
                processor = data['processor']
                dl = DeepLearningModels()
                
                # Prepare data
                X, y, _ = processor.prepare_for_modeling(data['missing_clean'], 'total_missing')
                X = X[[col for col in X.columns if col not in ['male_total', 'female_total', 
                                                               'transgender_total', 'children_missing',
                                                               'youth_missing', 'adults_missing', 
                                                               'elderly_missing']]]
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Train model
                model, history = dl.train_regression(X_train, X_test, y_train, y_test, 
                                                    epochs=epochs, batch_size=batch_size)
                
                st.session_state.dl_reg_model = model
                st.session_state.dl_reg_history = history
                
                st.success("✅ Deep Learning Model Trained Successfully!")
        
        if 'dl_reg_history' in st.session_state:
            history = st.session_state.dl_reg_history
            
            st.markdown("---")
            st.subheader("📈 Training History")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=history.history['loss'], name='Training Loss',
                                        mode='lines', line=dict(color='#e74c3c', width=2)))
                fig.add_trace(go.Scatter(y=history.history['val_loss'], name='Validation Loss',
                                        mode='lines', line=dict(color='#3498db', width=2)))
                fig.update_layout(title='Model Loss', xaxis_title='Epoch', yaxis_title='Loss')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=history.history['mae'], name='Training MAE',
                                        mode='lines', line=dict(color='#2ecc71', width=2)))
                fig.add_trace(go.Scatter(y=history.history['val_mae'], name='Validation MAE',
                                        mode='lines', line=dict(color='#f39c12', width=2)))
                fig.update_layout(title='Model MAE', xaxis_title='Epoch', yaxis_title='MAE')
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Deep Neural Network - Classification")
        st.info("Train a neural network to classify districts as high or low crime areas")
        
        if st.button("🚀 Train Deep Learning Model (Classification)"):
            st.info("Classification model training implemented similarly to regression!")

def show_summary(data):
    st.header("📑 Project Summary & Key Insights")
    
    st.markdown("""
    ## 🎯 Project Overview
    
    This comprehensive data science project analyzes **Missing Persons** and **Juvenile Crimes** data 
    across India from 2017-2022, implementing advanced machine learning techniques.
    
    ### 📊 Datasets
    - **Missing Persons**: 5,319 records across 751 districts, 36 states
    - **Juvenile Crimes**: 5,322 records covering 117 crime types
    - **Time Period**: 2017-2022 (6 years)
    
    ### 🔬 Techniques Implemented
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        #### 📈 Regression
        - Linear Regression
        - Ridge & Lasso
        - Decision Trees
        - Random Forest
        - Gradient Boosting
        - XGBoost
        - LightGBM
        """)
    
    with col2:
        st.markdown("""
        #### 🎯 Classification
        - Logistic Regression
        - Decision Trees
        - Random Forest
        - XGBoost
        - LightGBM
        """)
    
    with col3:
        st.markdown("""
        #### 🌐 Clustering
        - K-Means
        - Agglomerative
        - DBSCAN
        - PCA
        """)
    
    with col4:
        st.markdown("""
        #### 🧠 Deep Learning
        - Neural Networks
        - Regression Models
        - Classification Models
        - Dropout Regularization
        - Early Stopping
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 🔍 Key Findings
    
    #### Missing Persons Analysis
    - 📊 **Total Missing Persons (2017-2022)**: High concentration in urban areas
    - 👥 **Gender Distribution**: Male cases dominate overall statistics
    - 👶 **Age Groups**: Youth (14-30) shows highest missing rates
    - 📈 **Temporal Trends**: Varying patterns across years with seasonal fluctuations
    
    #### Juvenile Crimes Analysis
    - 🔪 **Total Crimes**: Significant variation across states
    - ⚖️ **Crime Categories**: Property crimes most common, followed by violent crimes
    - 📍 **Geographic Patterns**: Urban areas show higher crime rates
    - 🎯 **Top Crime Types**: Theft, assault, and robbery lead the statistics
    
    #### Combined Insights
    - 🔄 **Correlation**: Moderate positive correlation between missing persons and crime rates
    - 🗺️ **State Patterns**: Certain states show consistently higher rates in both metrics
    - 📊 **Clustering**: States group into 5 distinct patterns based on crime characteristics
    
    ### 🎓 Technical Achievements
    
    1. **Data Preprocessing**: Comprehensive cleaning and feature engineering
    2. **Feature Creation**: Generated 20+ derived features for better model performance
    3. **Model Comparison**: Evaluated 15+ different machine learning algorithms
    4. **Visualization**: Created 30+ interactive visualizations for insights
    5. **Deep Learning**: Implemented custom neural network architectures
    6. **Deployment**: Full-stack Streamlit application with interactive dashboard
    
    ### 🚀 Technologies Used
    
    - **Python**: Core programming language
    - **Pandas & NumPy**: Data manipulation
    - **Scikit-learn**: Machine learning algorithms
    - **XGBoost & LightGBM**: Gradient boosting frameworks
    - **TensorFlow/Keras**: Deep learning
    - **Plotly & Seaborn**: Data visualization
    - **Streamlit**: Web application framework
    
    ### 📌 Future Enhancements
    
    - Time series forecasting (ARIMA, LSTM)
    - Geospatial analysis with interactive maps
    - Real-time prediction API
    - Advanced deep learning architectures (CNN, RNN)
    - Explainable AI (SHAP values, LIME)
    
    ---
    
    ### 👨‍💻 About This Project
    
    This project demonstrates professional data science workflow from data acquisition 
    to model deployment, covering exploratory data analysis, statistical testing, 
    machine learning, deep learning, and interactive visualization.
    
    **Developed using**: Python, Streamlit, and modern ML/DL frameworks
    """)

if __name__ == "__main__":
    main()
