"""
Crime Analytics Dashboard - Interactive Analysis of Missing Persons and Juvenile Crimes
This Streamlit application provides comprehensive analysis and visualization of crime data
across Indian districts from 2017-2022.
"""

import os
import warnings
from typing import Tuple, Optional, List

import streamlit as st  # type: ignore
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import plotly.express as px  # type: ignore
from sklearn.cluster import KMeans  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
from sklearn.metrics import r2_score  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    page_title="Crime Analytics Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Page Constants ---
PAGE_HOME = "🏠 Home"
PAGE_EDA = "📊 Exploratory Data Analysis (EDA)"
PAGE_DISTRICT = "📍 District Deep Dive"
PAGE_MISSING = "🧍 Missing Persons Analysis"
PAGE_CRIME = "⚖️ Juvenile Crime Analysis"
PAGE_ADVANCED = "🔬 Advanced Analysis (Demo)"

# --- Chart Constants ---
CATEGORY_ORDER_ASCENDING = 'total ascending'
AGE_GROUP_LABEL = 'Age Group'
TOTAL_CASES_LABEL = 'Total Cases'
CRIME_TYPE_LABEL = 'Crime Type'

# --- Custom CSS ---
st.markdown("""
<style>
/* Main app background */
.stApp {
    background-color: #f0f2f6;
}

/* Page titles */
h1 {
    color: #1a1a1a;
    font-weight: 600;
}

/* Page subheaders */
h2 {
    color: #333333;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e0e0e0;
}

/* Metric boxes */
[data-testid="stMetric"] {
    background-color: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 15px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.04);
}
[data-testid="stMetricLabel"] {
    color: #555555;
    font-weight: 500;
}
[data-testid="stMetricValue"] {
    color: #007bff;
    font-size: 2.25rem;
    font-weight: 700;
}

/* Expander styling */
[data-testid="stExpander"] {
    background-color: #ffffff;
    border-radius: 10px;
    border: none;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04);
}
[data-testid="stExpander"] > summary {
    font-weight: 600;
    color: #007bff;
}

/* Chart container styling */
[data-testid="stPlotlyChart"] {
    background-color: #ffffff;
    border-radius: 10px;
    padding: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.04);
}

/* Dataframe styling */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.04);
}
</style>
""", unsafe_allow_html=True)

# --- Data Loading and Caching ---

@st.cache_data
def load_missing_persons_data(filepath: str) -> Tuple[Optional[pd.DataFrame], Optional[List[str]]]:
    """
    Loads and cleans the Missing Persons dataset.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        tuple: (DataFrame, list of numeric columns) or (None, None) on error
    """
    try:
        df = pd.read_csv(filepath, na_values=[""], low_memory=False)
        
        # Drop columns that are entirely null
        df = df.dropna(how='all', axis=1)
        
        # Identify gender/age columns
        male_cols = [col for col in df.columns if 'male_' in col.lower() and 'female' not in col.lower()]
        female_cols = [col for col in df.columns if 'female_' in col.lower()]
        trans_cols = [col for col in df.columns if 'trangender_' in col.lower() or 'transgender_' in col.lower()]
        
        all_numeric_cols = male_cols + female_cols + trans_cols
        
        # Convert all count columns to numeric, filling NaNs with 0
        for col in all_numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Create summary columns safely
        df['total_missing_male'] = df[male_cols].sum(axis=1) if male_cols else 0
        df['total_missing_female'] = df[female_cols].sum(axis=1) if female_cols else 0
        df['total_missing_trans'] = df[trans_cols].sum(axis=1) if trans_cols else 0
        df['total_missing'] = df['total_missing_male'] + df['total_missing_female'] + df['total_missing_trans']
        
        # Clean state/district names
        if 'state_name' in df.columns:
            df['state_name'] = df['state_name'].astype(str).str.strip().str.title()
        if 'district_name' in df.columns:
            df['district_name'] = df['district_name'].astype(str).str.strip().str.title()
        
        return df, all_numeric_cols
        
    except FileNotFoundError:
        st.error(f"❌ Error: The file '{filepath}' was not found. Please check the path.")
        return None, None
    except pd.errors.EmptyDataError:
        st.error(f"❌ Error: The file '{filepath}' is empty.")
        return None, None
    except Exception as e:
        st.error(f"❌ An error occurred while loading the missing persons data: {str(e)}")
        return None, None

@st.cache_data
def load_juvenile_crime_data(filepath: str) -> Tuple[Optional[pd.DataFrame], Optional[List[str]]]:
    """
    Loads and cleans the Juvenile IPC Crime dataset.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        tuple: (DataFrame, list of crime columns) or (None, None) on error
    """
    try:
        df = pd.read_csv(filepath, na_values=[""], low_memory=False)
        
        # Drop columns that are entirely null
        df = df.dropna(how='all', axis=1)
        
        # Identify non-crime columns
        id_cols = ['id', 'year', 'state_name', 'state_code', 'district_name', 'district_code', 'registration_circles']
        
        # Identify crime columns (all other columns)
        crime_cols = [col for col in df.columns if col not in id_cols]
        
        # Convert all crime columns to numeric, filling NaNs with 0
        for col in crime_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Create a total crimes column if not exists
        if 'total_ipc_crimes' not in df.columns:
            df['total_ipc_crimes'] = df[crime_cols].sum(axis=1)
        
        # Clean state/district names
        if 'state_name' in df.columns:
            df['state_name'] = df['state_name'].astype(str).str.strip().str.title()
        if 'district_name' in df.columns:
            df['district_name'] = df['district_name'].astype(str).str.strip().str.title()
        
        # Filter out summary rows
        if 'district_name' in df.columns:
            df = df[~df['district_name'].str.contains('Total', na=False, case=False)]
        
        return df, crime_cols
        
    except FileNotFoundError:
        st.error(f"❌ Error: The file '{filepath}' was not found. Please check the path.")
        return None, None
    except pd.errors.EmptyDataError:
        st.error(f"❌ Error: The file '{filepath}' is empty.")
        return None, None
    except Exception as e:
        st.error(f"❌ An error occurred while loading the juvenile crime data: {str(e)}")
        return None, None

# --- Load Data ---
# Dynamic file path detection for deployment compatibility

def get_data_file_path(filename: str) -> str:
    """Get the correct path for data files based on environment"""
    # Try current directory first
    if os.path.exists(filename):
        return filename
    
    # Try in data subdirectory
    data_path = os.path.join('data', filename)
    if os.path.exists(data_path):
        return data_path
    
    # Try in parent directory
    parent_path = os.path.join('..', filename)
    if os.path.exists(parent_path):
        return parent_path
    
    # Return the filename as is and let the error handler deal with it
    return filename

missing_persons_path = get_data_file_path('districtwise-missing-persons-2017-2022.csv')
juvenile_crime_path = get_data_file_path('districtwise-ipc-crime-by-juveniles-2017-onwards.csv')

# Now we pass these paths to the loading functions
missing_df_raw, missing_numeric_cols = load_missing_persons_data(missing_persons_path)
juvenile_df_raw, juvenile_crime_cols = load_juvenile_crime_data(juvenile_crime_path)


# --- Helper Functions ---

def get_gender_from_category(category: str) -> str:
    """Extract gender from category string."""
    if 'male_' in category and 'female' not in category:
        return 'Male'
    if 'female_' in category:
        return 'Female'
    return 'Transgender'

def clean_age_group(category: str) -> str:
    """Clean age group string."""
    return (category.replace('male_', '')
                   .replace('female_', '')
                   .replace('trangender_', '')
                   .replace('transgender_', '')
                   .replace('_', ' ')
                   .title())

@st.cache_data
def get_merged_data(_missing_df: Optional[pd.DataFrame], _juvenile_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Merges the two dataframes on year, state, and district.
    
    Args:
        _missing_df: Missing persons DataFrame
        _juvenile_df: Juvenile crimes DataFrame
        
    Returns:
        DataFrame: Merged data or empty DataFrame on error
    """
    if _missing_df is None or _juvenile_df is None:
        return pd.DataFrame()
    
    try:
        # Group by common keys to aggregate data before merging
        missing_agg = _missing_df.groupby(
            ['year', 'state_name', 'district_name'],
            as_index=False
        )['total_missing'].sum()
        
        juvenile_agg = _juvenile_df.groupby(
            ['year', 'state_name', 'district_name'],
            as_index=False
        )['total_ipc_crimes'].sum()
        
        merged_df = pd.merge(
            missing_agg,
            juvenile_agg,
            on=['year', 'state_name', 'district_name'],
            how='outer',
            validate='one_to_one'
        ).fillna(0)
        
        return merged_df
    except Exception as e:
        st.warning(f"Error merging datasets: {str(e)}")
        return pd.DataFrame()

# --- Sidebar Navigation ---
st.sidebar.title("Crime Analytics 🔍")
st.sidebar.markdown("Navigate through different analyses of missing persons and juvenile crime data from 2017-2022.")

if missing_df_raw is None or juvenile_df_raw is None:
    st.sidebar.error("Data loading failed. Please check file paths.")
else:
    page = st.sidebar.radio(
        "Select a Page",
        (PAGE_HOME, PAGE_EDA, PAGE_DISTRICT, PAGE_MISSING, PAGE_CRIME, PAGE_ADVANCED)
    )
    
    # --- Sidebar Filters ---
    st.sidebar.header("Global Filters")
    
    # Get unique sorted lists for filters
    all_states = sorted(pd.concat([missing_df_raw['state_name'], juvenile_df_raw['state_name']]).unique().tolist())
    all_years = sorted(pd.concat([missing_df_raw['year'], juvenile_df_raw['year']]).unique().tolist())
    
    # Year Filter
    selected_year = st.sidebar.selectbox(
        "Select Year",
        ["All"] + all_years,
        index=0
    )
    
    # State Filter
    selected_state = st.sidebar.selectbox(
        "Select State",
        ["All"] + all_states,
        index=0
    )
    
    # --- Filter Dataframes ---
    missing_df = missing_df_raw.copy()
    juvenile_df = juvenile_df_raw.copy()
    
    if selected_year != "All":
        missing_df = missing_df[missing_df['year'] == selected_year]
        juvenile_df = juvenile_df[juvenile_df['year'] == selected_year]
        
    if selected_state != "All":
        missing_df = missing_df[missing_df['state_name'] == selected_state]
        juvenile_df = juvenile_df[juvenile_df['state_name'] == selected_state]
        
    # --- Page Implementations ---

    def show_home_page() -> None:
        """Display the home page with overview statistics."""
        st.title("🏠 Welcome to the Crime Analytics Dashboard")
        st.markdown(f"""
        This dashboard provides an interactive analysis of **Missing Persons** and **Juvenile Crimes** across various districts from 2017 to 2022.
        
        Use the sidebar to navigate to different sections:
        - **📊 EDA**: Compare high-level trends between the two datasets.
        - **📍 District Deep Dive**: Focus on a single district's profile.
        - **🧍 Missing Persons Analysis**: Dive deep into missing persons statistics.
        - **⚖️ Juvenile Crime Analysis**: Explore trends in different IPC crime categories.
        - **🔬 Advanced Analysis**: See demo models for clustering and regression.
        
        **Data Filters Applied:**
        - **Year:** `{selected_year}`
        - **State:** `{selected_state}`
        """)
        
        st.divider()
        
        st.header("High-Level Statistics (Based on Filters)")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Missing Persons", f"{missing_df['total_missing'].sum():,}")
        col2.metric("Total Juvenile IPC Crimes", f"{juvenile_df['total_ipc_crimes'].sum():,}")
        
        # Safely get district count
        try:
            total_districts = pd.concat([missing_df['district_name'], juvenile_df['district_name']]).nunique()
            col3.metric("Districts/Circles Analyzed", f"{total_districts:,}")
        except Exception:
            col3.metric("Districts/Circles Analyzed", "N/A")

        st.divider()
        st.header("Data Over Time (All Years, All States)")
        
        # Prepare data for time series chart
        missing_time = missing_df_raw.groupby('year')['total_missing'].sum().reset_index()
        juvenile_time = juvenile_df_raw.groupby('year')['total_ipc_crimes'].sum().reset_index()
        
        time_df = pd.merge(missing_time, juvenile_time, on='year', how='outer', validate='one_to_one').fillna(0)
        time_df_melted = time_df.melt('year', var_name='Metric', value_name='Count')
        
        fig = px.line(
            time_df_melted,
            x='year',
            y='Count',
            color='Metric',
            title="Total Missing Persons vs. Total Juvenile Crimes",
            markers=True,
            labels={'Count': 'Total Count', 'year': 'Year'},
            color_discrete_map={
                'total_missing': '#007bff',
                'total_ipc_crimes': '#ff6347'
            }
        )
        fig.update_layout(legend_title_text='Metric')
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Show Raw Data Samples"):
            st.subheader("Missing Persons Data (Top 5 Rows)")
            st.dataframe(missing_df_raw.head())
            st.subheader("Juvenile Crime Data (Top 5 Rows)")
            st.dataframe(juvenile_df_raw.head())

    def show_eda_page() -> None:
        """Display exploratory data analysis page."""
        st.title("📊 Exploratory Data Analysis (EDA)")
        st.markdown("Comparing Missing Persons and Juvenile Crimes based on selected filters.")
        
        # Get filtered, merged data
        merged_df = get_merged_data(missing_df, juvenile_df)

        if merged_df.empty:
            st.warning("No data available for the selected filters to compare.")
            return

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Top 10 States by Total Missing Persons")
            # Group by state from the filtered data
            state_missing = missing_df.groupby('state_name')['total_missing'].sum().nlargest(10).reset_index()
            fig1 = px.bar(
                state_missing,
                x='total_missing',
                y='state_name',
                orientation='h',
                title="Top 10 States (Missing Persons)",
                labels={'total_missing': 'Total Missing', 'state_name': 'State'},
                color='total_missing',
                color_continuous_scale='Blues'
            )
            fig1.update_layout(yaxis={'categoryorder': CATEGORY_ORDER_ASCENDING})
            st.plotly_chart(fig1, use_container_width=True)
            
        with col2:
            st.subheader("Top 10 States by Total Juvenile Crimes")
            # Group by state from the filtered data
            state_juvenile = juvenile_df.groupby('state_name')['total_ipc_crimes'].sum().nlargest(10).reset_index()
            fig2 = px.bar(
                state_juvenile,
                x='total_ipc_crimes',
                y='state_name',
                orientation='h',
                title="Top 10 States (Juvenile Crimes)",
                color='total_ipc_crimes',
                color_continuous_scale='Reds',
                labels={'total_ipc_crimes': 'Total Juvenile Crimes', 'state_name': 'State'}
            )
            fig2.update_layout(yaxis={'categoryorder': CATEGORY_ORDER_ASCENDING})
            st.plotly_chart(fig2, use_container_width=True)
            
        st.divider()
        st.subheader("Missing Persons vs. Juvenile Crimes")
        st.markdown("Each point represents a district (or aggregated data).")
        
        # Aggregate by state if "All" districts is implied (i.e., state is not "All")
        if selected_state != "All":
            plot_data = merged_df.groupby('district_name')[['total_missing', 'total_ipc_crimes']].sum().reset_index()
            hover_name = 'district_name'
            title = f"Missing Persons vs. Juvenile Crimes by District in {selected_state}"
        else:
            plot_data = merged_df.groupby('state_name')[['total_missing', 'total_ipc_crimes']].sum().reset_index()
            hover_name = 'state_name'
            title = "Missing Persons vs. Juvenile Crimes by State"

        fig3 = px.scatter(
            plot_data,
            x='total_ipc_crimes',
            y='total_missing',
            title=title,
            hover_name=hover_name,
            trendline="ols",
            trendline_color_override="red",
            labels={
                'total_ipc_crimes': 'Total Juvenile IPC Crimes',
                'total_missing': 'Total Missing Persons'
            }
        )
        st.plotly_chart(fig3, use_container_width=True)

    def show_district_deep_dive_page() -> None:
        """
        Display detailed analysis of a single selected district.
        This page ignores the global sidebar filters to provide a focused view.
        """
        st.title("📍 District Deep Dive")
        st.markdown("Select a state and district to see a detailed breakdown. This page ignores the global filters in the sidebar.")
        
        # --- State and District Selectors ---
        try:
            all_states_list = sorted(pd.concat([missing_df_raw['state_name'], juvenile_df_raw['state_name']]).unique().tolist())
            selected_state_dd = st.selectbox(
                "Select State",
                all_states_list,
                index=0,
                key="dd_state_select"
            )
            
            if selected_state_dd:
                available_districts = sorted(pd.concat([
                    missing_df_raw[missing_df_raw['state_name'] == selected_state_dd]['district_name'],
                    juvenile_df_raw[juvenile_df_raw['state_name'] == selected_state_dd]['district_name']
                ]).unique().tolist())
                
                if not available_districts:
                    st.warning("No districts found for the selected state.")
                    return

                selected_district_dd = st.selectbox(
                    "Select District",
                    available_districts,
                    index=0,
                    key="dd_district_select"
                )
            else:
                st.info("Please select a state.")
                return

        except Exception as e:
            st.error(f"Error loading district filters: {e}")
            return
            
        if not selected_district_dd:
            st.info("Please select a district to begin analysis.")
            return

        st.header(f"Analytics for: {selected_district_dd}, {selected_state_dd}")
        
        # --- Filter Data for the selected district ---
        missing_district_df = missing_df_raw[
            (missing_df_raw['state_name'] == selected_state_dd) & 
            (missing_df_raw['district_name'] == selected_district_dd)
        ]
        juvenile_district_df = juvenile_df_raw[
            (juvenile_df_raw['state_name'] == selected_state_dd) &
            (juvenile_df_raw['district_name'] == selected_district_dd)
        ]

        if missing_district_df.empty and juvenile_district_df.empty:
            st.warning("No data available for the selected district.")
            return

        # --- KPIs ---
        st.subheader("Key Metrics (All Years)")
        col1, col2 = st.columns(2)
        col1.metric("Total Missing Persons", f"{missing_district_df['total_missing'].sum():,}")
        col2.metric("Total Juvenile IPC Crimes", f"{juvenile_district_df['total_ipc_crimes'].sum():,}")
        
        st.divider()
        
        # --- Time Series Plot ---
        st.subheader("Trends Over Time")
        missing_time = missing_district_df.groupby('year')['total_missing'].sum().reset_index()
        juvenile_time = juvenile_district_df.groupby('year')['total_ipc_crimes'].sum().reset_index()
        
        time_df = pd.merge(missing_time, juvenile_time, on='year', how='outer', validate='one_to_one').fillna(0)
        time_df_melted = time_df.melt('year', var_name='Metric', value_name='Count')
        
        if not time_df_melted.empty:
            fig_time = px.line(
                time_df_melted,
                x='year',
                y='Count',
                color='Metric',
                title=f"Missing Persons vs. Juvenile Crimes in {selected_district_dd}",
                markers=True,
                labels={'Count': 'Total Count', 'year': 'Year'},
                color_discrete_map={
                    'total_missing': '#007bff',
                    'total_ipc_crimes': '#ff6347'
                }
            )
            fig_time.update_layout(legend_title_text='Metric')
            st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.info("No time series data available for this district.")

        st.divider()
        # --- Detailed Breakdowns ---
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Missing Persons by Gender & Age")
            if not missing_district_df.empty:
                # Prepare data for stacked bar
                age_cols = [col for col in missing_numeric_cols if 'total_' not in col]
                age_df = missing_district_df[age_cols].sum().reset_index()
                age_df.columns = ['Category', 'Count']
                
                # Extract gender and age group
                age_df['Gender'] = age_df['Category'].apply(get_gender_from_category)
                age_df[AGE_GROUP_LABEL] = age_df['Category'].apply(clean_age_group)
                
                fig_age = px.bar(
                    age_df[age_df['Count'] > 0], # Plot only categories with data
                    x=AGE_GROUP_LABEL,
                    y='Count',
                    color='Gender',
                    title=f"Missing Persons in {selected_district_dd}",
                    barmode='group',
                    color_discrete_map={
                        'Male': '#007bff',
                        'Female': '#E83E8C',
                        'Transgender': '#6f42c1'
                    }
                )
                st.plotly_chart(fig_age, use_container_width=True)
            else:
                st.info("No Missing Persons data for this district.")

        with col2:
            st.subheader("Top 5 Juvenile Crime Types")
            if not juvenile_district_df.empty:
                # Use juvenile_crime_cols (excluding total)
                crime_cols_no_total = [col for col in juvenile_crime_cols if col != 'total_ipc_crimes']
                crime_data = juvenile_district_df[crime_cols_no_total].sum().nlargest(5).reset_index()
                crime_data.columns = [CRIME_TYPE_LABEL, TOTAL_CASES_LABEL]
                
                if not crime_data.empty and crime_data[TOTAL_CASES_LABEL].sum() > 0:
                    fig_crime = px.bar(
                        crime_data,
                        x=TOTAL_CASES_LABEL,
                        y=CRIME_TYPE_LABEL,
                        orientation='h',
                        title=f"Top 5 Juvenile Crimes in {selected_district_dd}",
                        color=TOTAL_CASES_LABEL,
                        color_continuous_scale='Oranges'
                    )
                    fig_crime.update_layout(yaxis={'categoryorder': CATEGORY_ORDER_ASCENDING})
                    st.plotly_chart(fig_crime, use_container_width=True)
                else:
                    st.info("No juvenile crime incidents recorded for this district.")
            else:
                st.info("No Juvenile Crime data for this district.")

    def show_missing_persons_page() -> None:
        """Display missing persons analysis page."""
        st.title("🧍 Missing Persons Analysis")
        
        if missing_df.empty:
            st.warning("No missing persons data available for the selected filters.")
            return

        # District Filter (specific to this page)
        available_districts = sorted(missing_df['district_name'].unique().tolist())
        selected_district = st.selectbox(
            "Select District (Optional)",
            ["All"] + available_districts,
            index=0
        )
        
        page_df = missing_df.copy()
        if selected_district != "All":
            page_df = page_df[page_df['district_name'] == selected_district]
            
        st.header(f"Showing Data for: {selected_state} | {selected_district} | {selected_year}")

        st.divider()
        # KPIs
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Missing", f"{page_df['total_missing'].sum():,}")
        col2.metric("Male Missing", f"{page_df['total_missing_male'].sum():,}")
        col3.metric("Female Missing", f"{page_df['total_missing_female'].sum():,}")
        col4.metric("Trans Missing", f"{page_df['total_missing_trans'].sum():,}")
        
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Missing Persons by Gender and Age Group")
            
            # Prepare data for stacked bar
            age_cols = [col for col in missing_numeric_cols if 'total_' not in col]
            age_df = page_df[age_cols].sum().reset_index()
            age_df.columns = ['Category', 'Count']
            
            # Extract gender and age group
            age_df['Gender'] = age_df['Category'].apply(get_gender_from_category)
            age_df[AGE_GROUP_LABEL] = age_df['Category'].apply(clean_age_group)
            
            fig1 = px.bar(
                age_df,
                x=AGE_GROUP_LABEL,
                y='Count',
                color='Gender',
                title="Missing Persons by Age Group and Gender",
                barmode='group',
                color_discrete_map={
                    'Male': '#007bff',
                    'Female': '#E83E8C',
                    'Transgender': '#6f42c1'
                }
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.subheader("Top 10 Districts (Filtered)")
            
            if selected_district == "All":
                top_districts = page_df.groupby('district_name')['total_missing'].sum().nlargest(10).reset_index()
                title = "Top 10 Districts by Missing Persons"
                fig2 = px.bar(
                    top_districts,
                    x='total_missing',
                    y='district_name',
                    orientation='h',
                    title=title,
                    labels={'total_missing': 'Total Missing', 'district_name': 'District'},
                    color='total_missing',
                    color_continuous_scale='Blues'
                )
                fig2.update_layout(yaxis={'categoryorder': CATEGORY_ORDER_ASCENDING})
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Displaying data for a single selected district.")
                st.dataframe(page_df[['district_name', 'total_missing', 'total_missing_male', 'total_missing_female']].head())


    def show_juvenile_crime_page() -> None:
        """Display juvenile crime analysis page."""
        st.title("⚖️ Juvenile Crime Analysis")
        
        if juvenile_df.empty:
            st.warning("No juvenile crime data available for the selected filters.")
            return

        # District Filter (specific to this page)
        available_districts = sorted(juvenile_df['district_name'].unique().tolist())
        selected_district = st.selectbox(
            "Select District (Optional)",
            ["All"] + available_districts,
            index=0
        )
        
        # Crime Type Filter
        # Select key crimes for easier filtering, add 'total_ipc_crimes'
        key_crimes = ['total_ipc_crimes', 'murder', 'hit_and_run', 'other_accidents', 'dowry_death', 'kidnpd_abdctn', 'rape', 'sexual_harassment_at_work', 'stalking', 'theft', 'riots', 'arson']
        # Ensure selected key crimes are actually in the columns
        available_key_crimes = [col for col in key_crimes if col in juvenile_df.columns]
        selected_crime = st.selectbox(
            "Select Crime Type to Analyze",
            available_key_crimes
        )
        
        page_df = juvenile_df.copy()
        if selected_district != "All":
            page_df = page_df[page_df['district_name'] == selected_district]
            
        st.header(f"Showing Data for: {selected_state} | {selected_district} | {selected_year}")
        
        st.divider()
        # KPIs
        col1, col2 = st.columns(2)
        col1.metric(f"Total '{selected_crime}' Cases", f"{page_df[selected_crime].sum():,}")
        if selected_crime != 'total_ipc_crimes':
            col2.metric("Total All Juvenile Crimes", f"{page_df['total_ipc_crimes'].sum():,}")

        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"Top 10 Districts for '{selected_crime}'")
            if selected_district == "All":
                top_districts = page_df.groupby('district_name')[selected_crime].sum().nlargest(10).reset_index()
                fig1 = px.bar(
                    top_districts,
                    x=selected_crime,
                    y='district_name',
                    orientation='h',
                    title=f"Top 10 Districts ({selected_crime})",
                    labels={selected_crime: f'Total {selected_crime}', 'district_name': 'District'},
                    color=selected_crime,
                    color_continuous_scale='Oranges'
                )
                fig1.update_layout(yaxis={'categoryorder': CATEGORY_ORDER_ASCENDING})
                st.plotly_chart(fig1, use_container_width=True)
            else:
                st.info(f"Displaying data for {selected_district}.")
                # --- FIX for ValueError ---
                # Create the list of columns to show
                columns_to_show = ['district_name', selected_crime, 'total_ipc_crimes']
                
                # Remove duplicates while preserving order
                unique_columns = list(dict.fromkeys(columns_to_show))
                
                st.dataframe(page_df[unique_columns].head())
                # --- END FIX ---
        
        with col2:
            st.subheader(f"Trend for '{selected_crime}' (All Years, Filtered State/District)")
            
            # Use raw (unfiltered by year) data for time series
            trend_df = juvenile_df_raw.copy()
            if selected_state != "All":
                trend_df = trend_df[trend_df['state_name'] == selected_state]
            if selected_district != "All":
                trend_df = trend_df[trend_df['district_name'] == selected_district]
                
            time_trend = trend_df.groupby('year')[selected_crime].sum().reset_index()
            
            fig2 = px.line(
                time_trend,
                x='year',
                y=selected_crime,
                title=f"Trend of '{selected_crime}' Over Time",
                markers=True,
                labels={selected_crime: 'Total Cases', 'year': 'Year'}
            )
            fig2.update_traces(line_color='#ff6347', line_width=3)
            st.plotly_chart(fig2, use_container_width=True)

        st.divider()
        # Correlation Heatmap
        st.subheader("Correlation Between Key Crime Types")
        st.markdown("This heatmap shows which crimes tend to occur together (based on filtered data).")
        
        if not page_df[available_key_crimes].empty:
            corr = page_df[available_key_crimes].corr()
            fig_corr = px.imshow(
                corr,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='RdBu_r',
                zmin=-1,
                zmax=1,
                title="Correlation Matrix of Key Juvenile Crimes"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.warning("Not enough data to calculate correlation for the current filters.")


    def show_advanced_analysis_page() -> None:
        """Display advanced analysis page with ML models."""
        st.title("🔬 Advanced Analysis (Demo)")
        st.markdown("""
        This page demonstrates some of the advanced analysis techniques mentioned in the `Crime_Analytics.ipynb` notebook,
        such as Clustering and Regression.
        """)
        
        merged_df = get_merged_data(missing_df_raw, juvenile_df_raw)
        
        # Aggregate by district across all years for a stable clustering
        cluster_data = merged_df.groupby(['state_name', 'district_name'])[['total_missing', 'total_ipc_crimes']].sum().reset_index()
        
        # Filter out districts with no activity
        cluster_data = cluster_data[(cluster_data['total_missing'] > 0) & (cluster_data['total_ipc_crimes'] > 0)]
        
        if cluster_data.empty or len(cluster_data) < 5: # Need enough data to cluster
            st.warning("Not enough data to perform advanced analysis. Try clearing filters.")
            return

        # --- 1. K-Means Clustering ---
        st.header("K-Means Clustering of Districts")
        st.markdown("""
        We can group districts into clusters based on their crime and missing person profiles.
        This helps identify "hotspots" or areas with similar characteristics.
        """)
        
        n_clusters = st.slider("Select number of clusters (K)", min_value=2, max_value=8, value=3)
        
        # Prepare data for clustering
        features = cluster_data[['total_missing', 'total_ipc_crimes']]
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Run K-Means
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_data['Cluster'] = kmeans.fit_predict(features_scaled)
            cluster_data['Cluster'] = cluster_data['Cluster'].astype(str) # For discrete colors
            
            fig_cluster = px.scatter(
                cluster_data,
                x='total_ipc_crimes',
                y='total_missing',
                color='Cluster',
                hover_name='district_name',
                hover_data=['state_name'],
                title=f"District Clusters (K={n_clusters})",
                labels={
                    'total_ipc_crimes': 'Total Juvenile Crimes (All Years)',
                    'total_missing': 'Total Missing Persons (All Years)'
                },
                color_discrete_sequence=px.colors.qualitative.Vivid
            )
            st.plotly_chart(fig_cluster, use_container_width=True)
        except Exception as e:
            st.error(f"An error occurred during clustering: {e}")

        st.divider()
        # --- 2. Regression Analysis ---
        st.header("Linear Regression (Demo)")
        st.markdown("""
        Can we predict the number of missing persons based on the number of juvenile crimes?
        A simple linear regression can help us understand the strength of this relationship.
        """)
        
        # Use the same aggregated data
        features_reg = cluster_data[['total_ipc_crimes']]
        target_reg = cluster_data['total_missing']
        
        model = LinearRegression()
        model.fit(features_reg, target_reg)
        predictions = model.predict(features_reg)
        
        r2 = r2_score(target_reg, predictions)
        
        st.metric("Model R-squared (R²)", f"{r2:.3f}")
        st.markdown(f"""
        - **Interpretation**: An R² of `{r2:.3f}` suggests that **{r2*100:.1f}%** of the variance in missing persons
          can be (linearly) explained by the number of juvenile crimes at the district level.
        - The chart below shows the same scatter plot as the EDA page, but it's built on data aggregated across all years.
        """)
        
        fig_reg = px.scatter(
            cluster_data,
            x='total_ipc_crimes',
            y='total_missing',
            title="Regression: Juvenile Crimes vs. Missing Persons",
            hover_name='district_name',
            trendline="ols",
            trendline_color_override="#E83E8C", # A nice pink/red
            labels={
                'total_ipc_crimes': 'Total Juvenile Crimes (All Years)',
                'total_missing': 'Total Missing Persons (All Years)'
            }
        )
        fig_reg.update_traces(marker={'color': '#007bff', 'opacity': 0.7})
        st.plotly_chart(fig_reg, use_container_width=True)


    # --- Main App Logic ---
    if __name__ == "__main__":
        if missing_df_raw is not None and juvenile_df_raw is not None:
            if page == PAGE_HOME:
                show_home_page()
            elif page == PAGE_EDA:
                show_eda_page()
            elif page == PAGE_DISTRICT:
                show_district_deep_dive_page()
            elif page == PAGE_MISSING:
                show_missing_persons_page()
            elif page == PAGE_CRIME:
                show_juvenile_crime_page()
            elif page == PAGE_ADVANCED:
                show_advanced_analysis_page()
        else:
            st.title("Error")
            st.error("Data could not be loaded. Please ensure the CSV files are in the correct location.")

