import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
from io import BytesIO
import os

st.set_page_config(
    page_title="Social Media Campaign Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #00d4ff;
        font-weight: bold;
        text-align: center;
        margin-bottom: 30px;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
    }
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #00d4ff;
        margin-bottom: 15px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .insight-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        border-left: 5px solid #00ff88;
        box-shadow: 0 4px 15px rgba(0, 255, 136, 0.1);
    }
    .warning-box {
        background: linear-gradient(135deg, #2d1a1a 0%, #421313 100%);
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        border-left: 5px solid #ff6b6b;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.1);
    }
    .recommendation-box {
        background: linear-gradient(135deg, #1a2e1a 0%, #133213 100%);
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        border-left: 5px solid #ffd166;
        box-shadow: 0 4px 15px rgba(255, 209, 102, 0.1);
    }
    .stMetric {
        background: rgba(30, 30, 46, 0.7);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #444;
    }
    .css-1d391kg, .stApp {
        background: #0f0f1e;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    .stDataFrame {
        background: rgba(30, 30, 46, 0.7);
        border-radius: 10px;
        border: 1px solid #444;
    }
    .plotly-chart {
        background: rgba(30, 30, 46, 0.5) !important;
        border-radius: 10px;
        padding: 10px;
    }
    .sub-header {
        color: #00d4ff;
        font-size: 1.8rem;
        margin-top: 30px;
        margin-bottom: 20px;
        border-bottom: 2px solid #00d4ff;
        padding-bottom: 10px;
    }
    .report-header {
        font-size: 2rem;
        color: #ffffff;
        text-align: center;
        background: linear-gradient(90deg, #1a1a2e 0%, #16213e 100%);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 30px;
        border: 2px solid #00d4ff;
    }
    .report-section {
        background: rgba(30, 30, 46, 0.7);
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        border-left: 5px solid #00d4ff;
    }
    .report-metric {
        font-size: 2rem;
        font-weight: bold;
        color: #00ff88;
        margin: 10px 0;
    }
    .badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        margin: 5px;
    }
    .badge-success {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        color: white;
    }
    .badge-warning {
        background: linear-gradient(135deg, #fdcb6e 0%, #e17055 100%);
        color: white;
    }
    .badge-danger {
        background: linear-gradient(135deg, #e17055 0%, #d63031 100%);
        color: white;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    .filter-indicator {
        background: rgba(30, 30, 46, 0.5);
        padding: 8px 12px;
        border-radius: 8px;
        margin-bottom: 10px;
        border-left: 4px solid #00d4ff;
        font-size: 0.85rem;
    }
    .upload-section {
        background: rgba(30, 30, 46, 0.7);
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        border: 1px solid #444;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'data_source' not in st.session_state:
    st.session_state.data_source = 'default'
if 'page' not in st.session_state:
    st.session_state.page = 'dashboard'

def process_data(df):
    """Process uploaded data"""
    try:
        required_columns = ['Start_Date', 'End_Date', 'Revenue', 'Budget_Spent', 'ROI', 'CPA', 'Conversions']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            return None

        df['Start_Date'] = pd.to_datetime(df['Start_Date'], format='%m/%d/%Y', errors='coerce')
        df['End_Date'] = pd.to_datetime(df['End_Date'], format='%m/%d/%Y', errors='coerce')

        date_mask = df['Start_Date'].isna() | df['End_Date'].isna()
        if date_mask.any():
            st.warning(f"Found {date_mask.sum()} rows with invalid dates. These will be filtered out.")
            df = df[~date_mask].copy()

        df['Month'] = df['Start_Date'].dt.strftime('%Y-%m')
        df['Quarter'] = df['Start_Date'].dt.quarter
        df['Year'] = df['Start_Date'].dt.year
        df['Campaign_Duration'] = (df['End_Date'] - df['Start_Date']).dt.days
        df['Profit'] = df['Revenue'] - df['Budget_Spent']

        df['Budget_Utilization'] = np.where(
            df['Total_Budget'] > 0,
            (df['Budget_Spent'] / df['Total_Budget']) * 100,
            np.where(df['Total_Budget'] == 0, 0, np.nan)
        )

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)

        return df

    except Exception as e:
        st.error(f"❌ Error processing data: {str(e)}")
        return None

@st.cache_data
def load_default_data():
    try:
        if os.path.exists('Cleaned_Social_Media_Advertising.csv'):
            df = pd.read_csv('Cleaned_Social_Media_Advertising.csv')
            return process_data(df)
        else:
            return None
    except Exception as e:
        st.error(f"❌ Error loading default data: {str(e)}")
        return None

def format_large_number(num):
    """Format large numbers with K, M, B suffixes"""
    if pd.isna(num) or num == 0:
        return "0"

    num = float(num)
    abs_num = abs(num)

    if abs_num >= 1_000_000_000:
        return f"${num / 1_000_000_000:.1f}B"
    elif abs_num >= 1_000_000:
        return f"${num / 1_000_000:.1f}M"
    elif abs_num >= 1_000:
        return f"${num / 1_000:.1f}K"
    elif abs_num >= 1:
        return f"${num:,.0f}"
    else:
        return f"${num:.2f}"

def format_large_number_no_symbol(num):
    """Format large numbers with K, M, B suffixes without $ symbol"""
    if pd.isna(num) or num == 0:
        return "0"

    num = float(num)
    abs_num = abs(num)

    if abs_num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.1f}B"
    elif abs_num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif abs_num >= 1_000:
        return f"{num / 1_000:.1f}K"
    elif abs_num >= 1:
        return f"{num:,.0f}"
    else:
        return f"{num:.2f}"

# Sidebar
st.sidebar.markdown("""
<style>
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.header("🎛️ Navigation")
col_nav1, col_nav2 = st.sidebar.columns(2)
with col_nav1:
    if st.button("📊 Dashboard", use_container_width=True):
        st.session_state.page = 'dashboard'
with col_nav2:
    if st.button("📋 Generate Report", use_container_width=True):
        st.session_state.page = 'report'

st.sidebar.markdown("---")
st.sidebar.header("📤 Upload Data")

uploaded_file = st.sidebar.file_uploader(
    "Choose a CSV file",
    type=['csv'],
    help="Upload your campaign data CSV file"
)

if uploaded_file is not None:
    try:
        df_uploaded = pd.read_csv(uploaded_file)
        processed_data = process_data(df_uploaded)

        if processed_data is not None:
            st.session_state.uploaded_data = processed_data
            st.session_state.data_source = 'uploaded'
            st.sidebar.success(f"✅ Successfully uploaded {len(processed_data)} records")

            st.sidebar.info(f"""
            **File Details:**
            - Name: {uploaded_file.name}
            - Size: {uploaded_file.size / 1024:.1f} KB
            - Columns: {len(processed_data.columns)}
            - Rows: {len(processed_data)}
            """)
    except Exception as e:
        st.sidebar.error(f"❌ Error uploading file: {str(e)}")

if st.sidebar.button("🔄 Use Default Data", use_container_width=True):
    default_data = load_default_data()
    if default_data is not None:
        st.session_state.uploaded_data = None
        st.session_state.data_source = 'default'
        st.sidebar.success("✅ Using default data file")
    else:
        st.sidebar.error("❌ Default data file not found")

st.sidebar.markdown("---")
st.sidebar.header("🔍 Campaign Filters")

# Load data based on source
if st.session_state.data_source == 'uploaded' and st.session_state.uploaded_data is not None:
    df = st.session_state.uploaded_data
elif st.session_state.data_source == 'default':
    df = load_default_data()
    if df is None:
        st.error("No data available. Please upload a data file.")
        st.stop()
else:
    df = load_default_data()
    if df is None:
        st.info("📤 Please upload a CSV file or use the default data")
        st.stop()

# Data source indicator
st.sidebar.markdown("---")
if st.session_state.data_source == 'uploaded':
    st.sidebar.markdown("""
    <div style="background: rgba(0, 212, 255, 0.1); padding: 10px; border-radius: 8px; border-left: 4px solid #00d4ff;">
        <strong>📂 Data Source:</strong> Uploaded File
    </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.markdown("""
    <div style="background: rgba(0, 255, 136, 0.1); padding: 10px; border-radius: 8px; border-left: 4px solid #00ff88;">
        <strong>📂 Data Source:</strong> Default File
    </div>
    """, unsafe_allow_html=True)

# Filters
all_campaigns = ['All'] + sorted(df['Campaign_Name'].dropna().unique().tolist())
all_platforms = ['All'] + sorted(df['Platform_Name'].dropna().unique().tolist())
all_interests = ['All'] + sorted(df['interest'].dropna().unique().tolist()) if 'interest' in df.columns else ['All']
all_locations = ['All'] + sorted(df['Location'].dropna().unique().tolist()) if 'Location' in df.columns else ['All']
all_age_groups = ['All'] + sorted(df['Age_Group'].dropna().unique().tolist()) if 'Age_Group' in df.columns else ['All']
all_genders = ['All'] + sorted(df['Gender'].dropna().unique().tolist()) if 'Gender' in df.columns else ['All']
all_objectives = ['All'] + sorted(df['Objective'].dropna().unique().tolist()) if 'Objective' in df.columns else ['All']
all_languages = ['All'] + sorted(df['Language'].dropna().unique().tolist()) if 'Language' in df.columns else ['All']
all_companies = ['All'] + sorted(df['Company_Name'].dropna().unique().tolist()) if 'Company_Name' in df.columns else ['All']
all_statuses = ['All'] + sorted(df['status'].dropna().unique().tolist()) if 'status' in df.columns else ['All']

selected_company = st.sidebar.multiselect('Select Company', all_companies, default=['All'])
selected_campaign = st.sidebar.selectbox('Select Campaign', all_campaigns)

df['Start_Date'] = pd.to_datetime(df['Start_Date'], errors='coerce')
df['End_Date'] = pd.to_datetime(df['End_Date'], errors='coerce')


df['Start_Date'] = df['Start_Date'].fillna(pd.Timestamp('2020-01-01'))
df['End_Date'] = df['End_Date'].fillna(pd.Timestamp.now())

start_date, end_date = st.sidebar.date_input("Select Date Range", [df['Start_Date'].min().date(), df['Start_Date'].max().date()])
selected_platform = st.sidebar.multiselect('Select Platform', all_platforms, default=['All'])
selected_interest = st.sidebar.multiselect('Select Interest', all_interests, default=['All'])
selected_location = st.sidebar.multiselect('Select Location', all_locations, default=['All'])
selected_age_group = st.sidebar.multiselect('Select Age Group', all_age_groups, default=['All'])
selected_gender = st.sidebar.multiselect('Select Gender', all_genders, default=['All'])
selected_objective = st.sidebar.multiselect('Select Campaign Objective', all_objectives, default=['All'])
selected_language = st.sidebar.multiselect('Select Language', all_languages, default=['All'])
selected_status = st.sidebar.multiselect('Select Status', all_statuses, default=['All'])

min_roi = st.sidebar.slider("Minimum ROI", float(df['ROI'].min()), float(df['ROI'].max()), 0.0, 0.5)
min_budget = st.sidebar.slider("Minimum Budget ($)", 0, int(df['Total_Budget'].max()) if 'Total_Budget' in df.columns else 1000, 0, 100)
max_budget = st.sidebar.slider("Maximum Budget ($)", 0, int(df['Total_Budget'].max()) if 'Total_Budget' in df.columns else 1000, 600, 50)

filtered_df = df.copy()

if selected_campaign != 'All':
    filtered_df = filtered_df[filtered_df['Campaign_Name'] == selected_campaign]

if 'All' not in selected_platform:
    filtered_df = filtered_df[filtered_df['Platform_Name'].isin(selected_platform)]

if 'All' not in selected_interest and 'interest' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['interest'].isin(selected_interest)]

if 'All' not in selected_location and 'Location' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['Location'].isin(selected_location)]

if 'All' not in selected_age_group and 'Age_Group' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['Age_Group'].isin(selected_age_group)]

if 'All' not in selected_gender and 'Gender' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['Gender'].isin(selected_gender)]

if 'All' not in selected_objective and 'Objective' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['Objective'].isin(selected_objective)]

if 'All' not in selected_language and 'Language' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['Language'].isin(selected_language)]

if 'All' not in selected_company and 'Company_Name' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['Company_Name'].isin(selected_company)]

if 'All' not in selected_status and 'status' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['status'].isin(selected_status)]

filtered_df = filtered_df[
    (filtered_df['Start_Date'] >= pd.Timestamp(start_date)) &
    (filtered_df['Start_Date'] <= pd.Timestamp(end_date)) &
    (filtered_df['ROI'] >= min_roi)
    ]

if 'Total_Budget' in filtered_df.columns:
    filtered_df = filtered_df[
        (filtered_df['Total_Budget'] >= min_budget) &
        (filtered_df['Total_Budget'] <= max_budget)
        ]

if len(filtered_df) == 0:
    st.warning("⚠️ **No data found for the selected filters!** Please adjust your filter criteria.")
    st.info("Showing all data instead...")
    filtered_df = df.copy()

def generate_comprehensive_report(df):
    report = f"""
# 📊 Comprehensive Campaign Performance Report
**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Report Period:** {start_date} to {end_date}
**Total Campaigns Analyzed:** {len(df):,}

## 📈 Executive Summary

### 🎯 Key Performance Indicators
- **Total Revenue Generated:** ${df['Revenue'].sum():,.0f}
- **Total Profit:** ${df['Profit'].sum():,.0f}
- **Total Budget Spent:** ${df['Budget_Spent'].sum():,.0f}
- **Average ROI:** {df['ROI'].mean():.2f}x
- **Overall Success Rate:** {(df['ROI'] > 1).mean() * 100:.1f}%
- **Total Conversions:** {df['Conversions'].sum():,.0f}
- **Average Conversion Rate:** {df['Conversion_Rate'].mean():.1f}% if 'Conversion_Rate' in df.columns else 'N/A'

## 🏆 Top Performers Analysis

### 🥇 Best Performing Platforms
"""

    platform_performance = df.groupby('Platform_Name').agg({
        'ROI': 'mean',
        'Revenue': 'sum',
        'Budget_Spent': 'sum',
        'Conversions': 'sum'
    }).sort_values('ROI', ascending=False)

    for idx, (platform, data) in enumerate(platform_performance.head(3).iterrows(), 1):
        report += f"""
**{idx}. {platform}**
- ROI: {data['ROI']:.2f}x
- Revenue: ${data['Revenue']:,.0f}
- Budget: ${data['Budget_Spent']:,.0f}
- Conversions: {data['Conversions']:,.0f}
"""

    report += """
### 🎯 Best Performing Campaign Objectives
"""

    if 'Objective' in df.columns:
        objective_performance = df.groupby('Objective').agg({
            'ROI': 'mean',
            'Revenue': 'sum',
            'Conversion_Rate': 'mean' if 'Conversion_Rate' in df.columns else 'ROI'
        }).sort_values('ROI', ascending=False)

        for obj, data in objective_performance.head(3).iterrows():
            report += f"""
**{obj}**
- Average ROI: {data['ROI']:.2f}x
- Total Revenue: ${data['Revenue']:,.0f}
"""

    report += """
### 🌍 Geographic Performance
"""

    if 'Location' in df.columns:
        location_performance = df.groupby('Location').agg({
            'ROI': 'mean',
            'Revenue': 'sum',
            'Profit': 'sum'
        }).sort_values('ROI', ascending=False)

        for loc, data in location_performance.head(3).iterrows():
            report += f"""
**{loc}**
- ROI: {data['ROI']:.2f}x
- Revenue: ${data['Revenue']:,.0f}
- Profit: ${data['Profit']:,.0f}
"""

    report += """
## 📊 Detailed Performance Metrics

### 💰 Financial Metrics
"""

    financial_metrics = {
        'Total Revenue': f"${df['Revenue'].sum():,.0f}",
        'Total Profit': f"${df['Profit'].sum():,.0f}",
        'Average ROI': f"{df['ROI'].mean():.2f}x",
        'Budget Utilization': f"{df['Budget_Utilization'].mean():.1f}%"
    }

    for metric, value in financial_metrics.items():
        report += f"- **{metric}:** {value}\n"

    report += """
### 🎯 Conversion Metrics
"""

    conversion_metrics = {
        'Total Conversions': f"{df['Conversions'].sum():,.0f}",
        'Average CPA': f"${df['CPA'].mean():.2f}",
        'Average CPC': f"${df['CPC'].mean():.2f}" if 'CPC' in df.columns else 'N/A',
        'Average CTR': f"{df['CTR'].mean():.2f}%" if 'CTR' in df.columns else 'N/A'
    }

    if 'Conversion_Rate' in df.columns:
        conversion_metrics['Average Conversion Rate'] = f"{df['Conversion_Rate'].mean():.1f}%"

    for metric, value in conversion_metrics.items():
        report += f"- **{metric}:** {value}\n"

    report += """
### 👥 Demographic Insights
"""

    if 'Age_Group' in df.columns:
        age_performance = df.groupby('Age_Group').agg({
            'ROI': 'mean',
            'Revenue': 'sum'
        }).sort_values('ROI', ascending=False)

        report += "\n**Age Group Performance:**\n"
        for age, data in age_performance.iterrows():
            report += f"- **{age}:** ROI: {data['ROI']:.2f}x, Revenue: ${data['Revenue']:,.0f}\n"

    if 'Gender' in df.columns:
        gender_performance = df.groupby('Gender').agg({
            'ROI': 'mean',
            'Conversion_Rate': 'mean' if 'Conversion_Rate' in df.columns else 'ROI'
        })

        report += "\n**Gender Performance:**\n"
        for gender, data in gender_performance.iterrows():
            report += f"- **{gender}:** ROI: {data['ROI']:.2f}x\n"

    report += """
## ⚠️ Areas Needing Attention

### 🚨 High Cost Areas
"""

    high_cpa = df.groupby('Platform_Name')['CPA'].mean().idxmax()
    high_cpa_value = df.groupby('Platform_Name')['CPA'].mean().max()

    report += f"""
- **Highest CPA:** {high_cpa} (${high_cpa_value:.2f})
"""

    if 'CPC' in df.columns:
        high_cpc = df.groupby('Platform_Name')['CPC'].mean().idxmax()
        high_cpc_value = df.groupby('Platform_Name')['CPC'].mean().max()
        report += f"- **Highest CPC:** {high_cpc} (${high_cpc_value:.2f})"

    if 'CTR' in df.columns:
        low_ctr = df.groupby('Platform_Name')['CTR'].mean().idxmin()
        low_ctr_value = df.groupby('Platform_Name')['CTR'].mean().min()
        report += f"""
- **Lowest CTR:** {low_ctr} ({low_ctr_value:.2f}%)"""

    low_roi = df.groupby('Platform_Name')['ROI'].mean().idxmin()
    low_roi_value = df.groupby('Platform_Name')['ROI'].mean().min()

    report += f"""
- **Lowest ROI:** {low_roi} ({low_roi_value:.2f}x)
"""

    report += """
## 💡 Strategic Recommendations

### 🎯 Immediate Actions
"""

    best_platform = df.groupby('Platform_Name')['ROI'].mean().idxmax()
    best_location = df.groupby('Location')['ROI'].mean().idxmax() if 'Location' in df.columns else 'All'
    best_objective = df.groupby('Objective')['ROI'].mean().idxmax() if 'Objective' in df.columns else 'All'

    report += f"""
1. **Increase Investment in Top Platform:** Allocate 20% more budget to {best_platform}
2. **Expand in Best Location:** Focus additional campaigns in {best_location}
3. **Optimize Campaign Objectives:** Prioritize {best_objective} campaigns
4. **Reduce Spend on Low Performers:** Decrease budget for {low_roi} by 30%
"""

    if 'CTR' in df.columns:
        report += f"5. **Improve CTR:** A/B test ad creatives on {low_ctr}"

    report += """
### 📈 Long-term Strategies
1. **Diversify Platform Portfolio:** Test new platforms with similar audience
2. **Audience Segmentation:** Create hyper-targeted campaigns based on demographics
3. **Content Optimization:** Develop platform-specific content strategies
4. **Budget Allocation:** Implement ROI-based budget allocation model
5. **Performance Monitoring:** Set up weekly performance review meetings

## 📅 Timeline Analysis
**Campaigns by Month:** {len(df['Month'].unique())} months analyzed
**Average Campaign Duration:** {df['Campaign_Duration'].mean():.0f} days
**Peak Performance Month:** {df.groupby('Month')['ROI'].mean().idxmax()}
**Highest Revenue Month:** {df.groupby('Month')['Revenue'].sum().idxmax()}

*Report generated by Social Media Campaign Analytics System*
"""

    return report

# Dashboard Page
if st.session_state.page == 'dashboard':
    st.markdown('<h1 class="main-header">📊 Social Media Campaign Performance Dashboard</h1>', unsafe_allow_html=True)

    st.markdown("## 📈 Executive Summary")
    st.markdown("---")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        total_revenue = filtered_df['Revenue'].sum()
        st.metric(label="Total Revenue", value=format_large_number(total_revenue))

    with col2:
        total_profit = filtered_df['Profit'].sum()
        st.metric(label="Total Profit", value=format_large_number(total_profit))

    with col3:
        total_spent = filtered_df['Budget_Spent'].sum()
        st.metric(label="Total Budget Spent", value=format_large_number(total_spent))

    with col4:
        total_conversions = filtered_df['Conversions'].sum()
        st.metric(label="Total Conversions", value=format_large_number_no_symbol(total_conversions))

    with col5:
        if 'Conversion_Rate' in filtered_df.columns:
            avg_conversion_rate = filtered_df['Conversion_Rate'].mean()
            st.metric(label="Avg Conversion Rate", value=f"{avg_conversion_rate:.1f}%")
        else:
            avg_roi = filtered_df['ROI'].mean()
            st.metric(label="Average ROI", value=f"{avg_roi:.2f}x")

    col6, col7, col8, col9 = st.columns(4)

    with col6:
        avg_roi = filtered_df['ROI'].mean()
        st.metric(label="Average ROI", value=f"{avg_roi:.2f}x")

    with col7:
        avg_cpa = filtered_df['CPA'].mean()
        st.metric(label="Avg CPA", value=f"${avg_cpa:.2f}")

    with col8:
        if 'CPC' in filtered_df.columns:
            avg_cpc = filtered_df['CPC'].mean()
            st.metric(label="Avg CPC", value=f"${avg_cpc:.2f}")
        else:
            total_profit = filtered_df['Profit'].sum()
            st.metric(label="Total Profit", value=format_large_number(total_profit))

    with col9:
        if 'CTR' in filtered_df.columns:
            avg_ctr = filtered_df['CTR'].mean()
            st.metric(label="Avg CTR", value=f"{avg_ctr:.2f}%")
        else:
            success_rate = (filtered_df['ROI'] > 1).mean() * 100
            st.metric(label="Success Rate", value=f"{success_rate:.1f}%")

    st.markdown("---")

    st.markdown('<h2 class="sub-header" style="color: #9d4edd;">📈 Performance Analysis</h2>', unsafe_allow_html=True)

    st.markdown("#### 📊 Platform Performance Analysis")

    platform_row1_col1, platform_row1_col2 = st.columns(2)

    with platform_row1_col1:
        platform_comparison = filtered_df.groupby('Platform_Name').agg({
            'Revenue': 'sum',
            'Budget_Spent': 'sum',
            'Conversions': 'sum',
            'ROI': 'mean'
        }).reset_index()

        fig_platform_comparison = go.Figure()

        fig_platform_comparison.add_trace(go.Bar(
            x=platform_comparison['Platform_Name'],
            y=platform_comparison['Revenue'],
            name='Revenue',
            marker_color='#1e40af',
            text=platform_comparison['Revenue'].apply(lambda x: f"${x:,.0f}"),
            textposition='outside'
        ))

        fig_platform_comparison.add_trace(go.Bar(
            x=platform_comparison['Platform_Name'],
            y=platform_comparison['Budget_Spent'],
            name='Budget Spent',
            marker_color='#ec4899',
            text=platform_comparison['Budget_Spent'].apply(lambda x: f"${x:,.0f}"),
            textposition='outside'
        ))

        fig_platform_comparison.add_trace(go.Scatter(
            x=platform_comparison['Platform_Name'],
            y=platform_comparison['ROI'],
            name='ROI',
            mode='lines+markers',
            line=dict(color='#00ff88', width=3),
            marker=dict(size=10, symbol='diamond'),
            yaxis='y2'
        ))

        fig_platform_comparison.update_layout(
            title='📊 Platform Performance Comparison',
            template='plotly_dark',
            xaxis_title='Platform',
            yaxis=dict(title='Revenue/Budget ($)', side='left', tickformat='$,.0f'),
            yaxis2=dict(title='ROI (x)', side='right', overlaying='y', showgrid=False),
            barmode='group',
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig_platform_comparison, use_container_width=True)

    with platform_row1_col2:
        if 'CPC' in filtered_df.columns and 'CTR' in filtered_df.columns:
            platform_costs = filtered_df.groupby('Platform_Name').agg({
                'CPA': 'mean',
                'CPC': 'mean',
                'CTR': 'mean'
            }).reset_index()

            fig_platform_costs = go.Figure()

            fig_platform_costs.add_trace(go.Bar(
                x=platform_costs['Platform_Name'],
                y=platform_costs['CPA'],
                name='CPA',
                marker_color='#ef4444',
                text=platform_costs['CPA'].round(2),
                textposition='outside'
            ))

            fig_platform_costs.add_trace(go.Bar(
                x=platform_costs['Platform_Name'],
                y=platform_costs['CPC'],
                name='CPC',
                marker_color='#f97316',
                text=platform_costs['CPC'].round(2),
                textposition='outside'
            ))

            fig_platform_costs.add_trace(go.Scatter(
                x=platform_costs['Platform_Name'],
                y=platform_costs['CTR'],
                name='CTR (%)',
                mode='lines+markers',
                line=dict(color='#10b981', width=3),
                marker=dict(size=10, symbol='square'),
                yaxis='y2'
            ))

            fig_platform_costs.update_layout(
                title='💵 Cost Metrics by Platform (CPA, CPC, CTR)',
                template='plotly_dark',
                xaxis_title='Platform',
                yaxis=dict(title='Cost ($)', side='left', tickformat='$.2f'),
                yaxis2=dict(title='CTR (%)', side='right', overlaying='y', showgrid=False),
                barmode='group',
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

        else:
            platform_costs = filtered_df.groupby('Platform_Name').agg({
                'CPA': 'mean',
                'ROI': 'mean'
            }).reset_index()

            fig_platform_costs = go.Figure()

            fig_platform_costs.add_trace(go.Bar(
                x=platform_costs['Platform_Name'],
                y=platform_costs['CPA'],
                name='CPA',
                marker_color='#ef4444',
                text=platform_costs['CPA'].round(2),
                textposition='outside'
            ))

            fig_platform_costs.add_trace(go.Scatter(
                x=platform_costs['Platform_Name'],
                y=platform_costs['ROI'],
                name='ROI',
                mode='lines+markers',
                line=dict(color='#10b981', width=3),
                marker=dict(size=10, symbol='square'),
                yaxis='y2'
            ))

            fig_platform_costs.update_layout(
                title='💵 Cost vs ROI by Platform',
                template='plotly_dark',
                xaxis_title='Platform',
                yaxis=dict(title='CPA ($)', side='left', tickformat='$.2f'),
                yaxis2=dict(title='ROI (x)', side='right', overlaying='y', showgrid=False),
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                hovermode='x unified'
            )

        st.plotly_chart(fig_platform_costs, use_container_width=True)

    st.markdown("#### 📈 Financial Trends Analysis")

    trend_row1_col1, trend_row1_col2 = st.columns(2)

    with trend_row1_col1:
        monthly_financial = filtered_df.groupby('Month').agg({
            'Revenue': 'sum',
            'Profit': 'sum',
            'Budget_Spent': 'sum'
        }).reset_index()

        fig_financial_standalone = go.Figure()

        fig_financial_standalone.add_trace(go.Bar(
            x=monthly_financial['Month'],
            y=monthly_financial['Revenue'],
            name='Revenue',
            marker_color='#1e40af',
            text=monthly_financial['Revenue'].apply(lambda x: f"${x:,.0f}"),
            textposition='outside'
        ))

        fig_financial_standalone.add_trace(go.Bar(
            x=monthly_financial['Month'],
            y=monthly_financial['Profit'],
            name='Profit',
            marker_color='#00ff88',
            text=monthly_financial['Profit'].apply(lambda x: f"${x:,.0f}"),
            textposition='outside'
        ))

        fig_financial_standalone.add_trace(go.Scatter(
            x=monthly_financial['Month'],
            y=monthly_financial['Budget_Spent'],
            name='Budget Spent',
            mode='lines+markers',
            line=dict(color='#ff6b6b', width=3),
            marker=dict(size=8, symbol='circle'),
            yaxis='y2'
        ))

        fig_financial_standalone.update_layout(
            title='📈 Monthly Revenue, Profit & Budget Trend',
            template='plotly_dark',
            xaxis_title='Month',
            yaxis=dict(title='Revenue/Profit ($)', side='left', tickformat='$,.0f'),
            yaxis2=dict(title='Budget Spent ($)', side='right', overlaying='y', showgrid=False, tickformat='$,.0f'),
            barmode='group',
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig_financial_standalone, use_container_width=True)

    with trend_row1_col2:
        if 'Total_Budget' in filtered_df.columns and 'Expected_Budget' in filtered_df.columns:
            budget_analysis = filtered_df.groupby('Month').agg({
                'Total_Budget': 'sum',
                'Budget_Spent': 'sum',
                'Expected_Budget': 'sum'
            }).reset_index()

            fig_budget_analysis = go.Figure()

            fig_budget_analysis.add_trace(go.Bar(
                x=budget_analysis['Month'],
                y=budget_analysis['Total_Budget'],
                name='Planned Budget',
                marker_color='#1e40af',
                text=budget_analysis['Total_Budget'].apply(lambda x: f"${x:,.0f}"),
                textposition='outside'
            ))

            fig_budget_analysis.add_trace(go.Bar(
                x=budget_analysis['Month'],
                y=budget_analysis['Budget_Spent'],
                name='Actual Spent',
                marker_color='#ec4899',
                text=budget_analysis['Budget_Spent'].apply(lambda x: f"${x:,.0f}"),
                textposition='outside'
            ))

            fig_budget_analysis.update_layout(
                title='💰 Budget vs Planned Analysis',
                template='plotly_dark',
                xaxis_title='Month',
                yaxis=dict(title='Budget Amount ($)', side='left', tickformat='$,.0f'),
                yaxis2=dict(title='Expected Budget ($)', side='right', overlaying='y', showgrid=False, tickformat='$,.0f'),
                barmode='group',
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            st.plotly_chart(fig_budget_analysis, use_container_width=True)
        else:
            monthly_revenue = filtered_df.groupby('Month').agg({
                'Revenue': 'sum',
                'Budget_Spent': 'sum'
            }).reset_index()

            fig_revenue_budget = go.Figure()

            fig_revenue_budget.add_trace(go.Bar(
                x=monthly_revenue['Month'],
                y=monthly_revenue['Revenue'],
                name='Revenue',
                marker_color='#1e40af',
                text=monthly_revenue['Revenue'].apply(lambda x: f"${x:,.0f}"),
                textposition='outside'
            ))

            fig_revenue_budget.add_trace(go.Bar(
                x=monthly_revenue['Month'],
                y=monthly_revenue['Budget_Spent'],
                name='Budget Spent',
                marker_color='#ec4899',
                text=monthly_revenue['Budget_Spent'].apply(lambda x: f"${x:,.0f}"),
                textposition='outside'
            ))

            fig_revenue_budget.update_layout(
                title='📊 Monthly Revenue vs Budget',
                template='plotly_dark',
                xaxis_title='Month',
                yaxis_title='Amount ($)',
                yaxis=dict(tickformat='$,.0f'),
                barmode='group',
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )

            st.plotly_chart(fig_revenue_budget, use_container_width=True)

    st.markdown("#### 🗣️ Language & Campaign Status Analysis")

    trend_row2_col1, trend_row2_col2 = st.columns(2)

    with trend_row2_col1:
        if 'Language' in filtered_df.columns:
            language_data = filtered_df.copy()
            language_data['Revenue'] = language_data['Revenue'].fillna(0)
            language_data['ROI'] = language_data['ROI'].fillna(0)

            language_stats = language_data.groupby('Language').agg({
                'Campaign_Name': 'count',
                'Revenue': 'sum',
                'ROI': 'mean'
            }).reset_index()

            language_stats = language_stats.rename(columns={'Campaign_Name': 'Campaign_Count'})

            fig_language_treemap = px.treemap(
                language_stats,
                path=['Language'],
                values='Campaign_Count',
                color='Revenue',
                color_continuous_scale='RdYlGn',
                title='🗣️ Language Distribution',
                hover_data=['Campaign_Count', 'Revenue', 'ROI']
            )

            fig_language_treemap.update_layout(
                template='plotly_dark',
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )

            fig_language_treemap.update_traces(
                texttemplate="<b>%{label}</b><br>%{value} campaigns",
                textposition="middle center",
                hovertemplate="<b>%{label}</b><br>Campaigns: %{customdata[0]}<br>Revenue: $%{customdata[1]:,.0f}<br>ROI: %{customdata[2]:.2f}x"
            )

            st.plotly_chart(fig_language_treemap, use_container_width=True)
        else:
            st.info("Language data not available.")

    with trend_row2_col2:
        if 'status' in filtered_df.columns:
            status_counts = filtered_df['status'].value_counts().reset_index()
            status_counts.columns = ['Status', 'Count']

            fig_status_pie = px.pie(
                status_counts,
                values='Count',
                names='Status',
                title='📊 Campaign Status Distribution',
                color_discrete_sequence=px.colors.qualitative.Set3,
                hole=0.3
            )

            fig_status_pie.update_layout(
                template='plotly_dark',
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation="v", yanchor="top", y=0.95, xanchor="left", x=1.05)
            )

            fig_status_pie.update_traces(
                textposition='inside',
                textinfo='percent+label+value',
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}"
            )

            st.plotly_chart(fig_status_pie, use_container_width=True)
        else:
            st.info("Status data not available.")

    if 'Location' in filtered_df.columns:
        st.markdown('<h2 class="sub-header" style="color: #00ff88;">🗺️ Geographic Performance</h2>', unsafe_allow_html=True)

        geo_tab1, geo_tab2, geo_tab3, geo_tab4, geo_tab5 = st.tabs([
            "📊 Revenue by Location",
            "📱 Platform Distribution",
            "🎯 Interest Categories",
            "🎯 Campaign Objectives",
            "🗣️ Language Performance"
        ])

        with geo_tab1:
            location_analysis = filtered_df.groupby('Location').agg({
                'Revenue': 'sum',
                'Profit': 'sum',
                'ROI': 'mean',
                'Budget_Spent': 'sum'
            }).reset_index()

            fig_location_revenue = px.bar(
                location_analysis.sort_values('Revenue', ascending=False),
                x='Location',
                y='Revenue',
                color='Revenue',
                color_continuous_scale='Plasma',
                title='💰 Revenue by Location',
                text='Revenue'
            )

            fig_location_revenue.update_layout(
                template='plotly_dark',
                xaxis_title='Location',
                yaxis_title='Revenue ($)',
                yaxis=dict(tickformat='$,.0f'),
                height=450,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis_tickangle=-45
            )

            fig_location_revenue.update_traces(
                texttemplate='$%{text:,.0f}',
                textposition='outside'
            )

            st.plotly_chart(fig_location_revenue, use_container_width=True)

        with geo_tab2:
            platform_by_location = filtered_df.groupby(['Location', 'Platform_Name']).agg({
                'Revenue': 'sum',
                'ROI': 'mean',
                'campaign_ID': 'count'
            }).reset_index()

            platform_by_location = platform_by_location.rename(columns={'campaign_ID': 'Campaign_Count'})

            location_totals = platform_by_location.groupby('Location')['Campaign_Count'].transform('sum')
            platform_by_location['Percentage'] = (platform_by_location['Campaign_Count'] / location_totals * 100).round(1)

            fig_platform_map = px.bar(
                platform_by_location,
                x='Location',
                y='Campaign_Count',
                color='Platform_Name',
                title='📱 Platform Distribution by Location',
                text='Percentage',
                hover_data=['Revenue', 'ROI'],
                color_discrete_sequence=px.colors.qualitative.Set3
            )

            fig_platform_map.update_layout(
                template='plotly_dark',
                xaxis_title='Location',
                yaxis_title='Number of Campaigns',
                height=450,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis_tickangle=-45
            )

            fig_platform_map.update_traces(
                texttemplate='%{text}%',
                textposition='inside'
            )

            st.plotly_chart(fig_platform_map, use_container_width=True)

        with geo_tab3:
            if 'interest' in filtered_df.columns:
                interest_by_location = filtered_df.groupby(['Location', 'interest']).agg({
                    'Revenue': 'sum',
                    'ROI': 'mean',
                    'Campaign_Name': 'count'
                }).reset_index()

                interest_by_location = interest_by_location.rename(columns={'Campaign_Name': 'Campaign_Count'})

                top_interests = interest_by_location.sort_values(['Location', 'Revenue'], ascending=[True, False])
                top_interests = top_interests.groupby('Location').head(3)

                fig_interest_map = px.bar(
                    top_interests,
                    x='Location',
                    y='Revenue',
                    color='interest',
                    title='🎯 Top Interests by Location (Revenue)',
                    text='Revenue',
                    hover_data=['ROI', 'Campaign_Count'],
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )

                fig_interest_map.update_layout(
                    template='plotly_dark',
                    xaxis_title='Location',
                    yaxis_title='Revenue ($)',
                    yaxis=dict(tickformat='$,.0f'),
                    height=450,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis_tickangle=-45
                )

                fig_interest_map.update_traces(
                    texttemplate='$%{text:,.0f}',
                    textposition='outside'
                )

                st.plotly_chart(fig_interest_map, use_container_width=True)
            else:
                st.info("Interest data not available for this dataset.")

        with geo_tab4:
            if 'Objective' in filtered_df.columns:
                objective_by_location = filtered_df.groupby(['Location', 'Objective']).agg({
                    'Revenue': 'sum',
                    'ROI': 'mean',
                    'Budget_Spent': 'sum'
                }).reset_index()

                objective_by_location['Efficiency'] = objective_by_location['Revenue'] / objective_by_location['Budget_Spent'].replace(0, 1)

                fig_objective_map = px.scatter(
                    objective_by_location,
                    x='Location',
                    y='Revenue',
                    size='Budget_Spent',
                    color='Objective',
                    hover_name='Objective',
                    title='🎯 Objectives: Revenue vs Budget by Location',
                    labels={
                        'Location': 'Location',
                        'Revenue': 'Revenue ($)',
                        'Budget_Spent': 'Budget Spent ($)',
                        'Efficiency': 'Efficiency (Rev/Budget)'
                    },
                    size_max=40,
                    color_discrete_sequence=px.colors.qualitative.Bold
                )

                fig_objective_map.update_layout(
                    template='plotly_dark',
                    xaxis_title='Location',
                    yaxis_title='Revenue ($)',
                    yaxis=dict(tickformat='$,.0f'),
                    height=450,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis_tickangle=-45
                )

                st.plotly_chart(fig_objective_map, use_container_width=True)
            else:
                st.info("Objective data not available for this dataset.")

        with geo_tab5:
            if 'Language' in filtered_df.columns:
                language_by_location = filtered_df.groupby(['Location', 'Language']).agg({
                    'Revenue': 'sum',
                    'ROI': 'mean',
                    'Conversion_Rate': 'mean' if 'Conversion_Rate' in filtered_df.columns else 'ROI',
                    'Campaign_Name': 'count'
                }).reset_index()

                language_by_location = language_by_location.rename(columns={'Campaign_Name': 'Campaign_Count'})

                fig_language_map = px.sunburst(
                    language_by_location,
                    path=['Location', 'Language'],
                    values='Revenue',
                    color='ROI',
                    color_continuous_scale='RdYlGn',
                    title='🗣️ Language Distribution & Performance by Location',
                    hover_data=['Conversion_Rate' if 'Conversion_Rate' in filtered_df.columns else 'ROI', 'Campaign_Count']
                )

                fig_language_map.update_layout(
                    template='plotly_dark',
                    height=500,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )

                st.plotly_chart(fig_language_map, use_container_width=True)
            else:
                st.info("Language data not available for this dataset.")

    if 'Age_Group' in filtered_df.columns or 'Gender' in filtered_df.columns or 'interest' in filtered_df.columns:
        st.markdown('<h2 class="sub-header" style="color: #ff6b6b;">👥 Demographic Performance</h2>', unsafe_allow_html=True)

        demo_col1, demo_col2, demo_col3 = st.columns(3)

        with demo_col1:
            if 'Age_Group' in filtered_df.columns:
                age_performance = filtered_df.groupby('Age_Group').agg({
                    'Revenue': 'sum',
                    'Budget_Spent': 'sum',
                    'ROI': 'mean'
                }).reset_index()

                fig_age = go.Figure()

                fig_age.add_trace(go.Bar(
                    x=age_performance['Age_Group'],
                    y=age_performance['Revenue'],
                    name='Revenue',
                    marker_color='#1e40af',
                    text=age_performance['Revenue'].apply(lambda x: f"${x:,.0f}"),
                    textposition='outside'
                ))

                fig_age.add_trace(go.Bar(
                    x=age_performance['Age_Group'],
                    y=age_performance['Budget_Spent'],
                    name='Budget Spent',
                    marker_color='#ec4899',
                    text=age_performance['Budget_Spent'].apply(lambda x: f"${x:,.0f}"),
                    textposition='outside'
                ))

                fig_age.update_layout(
                    title='💰 Revenue vs Budget Spent by Age Group',
                    template='plotly_dark',
                    xaxis_title='Age Group',
                    yaxis_title='Amount ($)',
                    yaxis=dict(tickformat='$,.0f'),
                    barmode='group',
                    height=350,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )

                st.plotly_chart(fig_age, use_container_width=True)
            else:
                st.info("Age Group data not available.")

        with demo_col2:
            if 'Gender' in filtered_df.columns:
                gender_performance = filtered_df.groupby('Gender').agg({
                    'Revenue': 'sum',
                    'Budget_Spent': 'sum',
                    'ROI': 'mean'
                }).reset_index()

                fig_gender = go.Figure()

                fig_gender.add_trace(go.Bar(
                    x=gender_performance['Gender'],
                    y=gender_performance['Revenue'],
                    name='Revenue',
                    marker_color='#1e40af',
                    text=gender_performance['Revenue'].apply(lambda x: f"${x:,.0f}"),
                    textposition='outside'
                ))

                fig_gender.add_trace(go.Bar(
                    x=gender_performance['Gender'],
                    y=gender_performance['Budget_Spent'],
                    name='Budget Spent',
                    marker_color='#ec4899',
                    text=gender_performance['Budget_Spent'].apply(lambda x: f"${x:,.0f}"),
                    textposition='outside'
                ))

                fig_gender.update_layout(
                    title='💰 Revenue vs Budget Spent by Gender',
                    template='plotly_dark',
                    xaxis_title='Gender',
                    yaxis_title='Amount ($)',
                    yaxis=dict(tickformat='$,.0f'),
                    barmode='group',
                    height=350,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )

                st.plotly_chart(fig_gender, use_container_width=True)
            else:
                st.info("Gender data not available.")

        with demo_col3:
            if 'interest' in filtered_df.columns:
                interest_performance = filtered_df.groupby('interest').agg({
                    'Revenue': 'sum',
                    'Budget_Spent': 'sum',
                    'ROI': 'mean'
                }).reset_index().sort_values('Revenue', ascending=False)

                fig_interest = go.Figure()

                fig_interest.add_trace(go.Bar(
                    x=interest_performance.head(10)['interest'],
                    y=interest_performance.head(10)['Revenue'],
                    name='Revenue',
                    marker_color='#1e40af',
                    text=interest_performance.head(10)['Revenue'].apply(lambda x: f"${x:,.0f}"),
                    textposition='outside'
                ))

                fig_interest.add_trace(go.Bar(
                    x=interest_performance.head(10)['interest'],
                    y=interest_performance.head(10)['Budget_Spent'],
                    name='Budget Spent',
                    marker_color='#ec4899',
                    text=interest_performance.head(10)['Budget_Spent'].apply(lambda x: f"${x:,.0f}"),
                    textposition='outside'
                ))

                fig_interest.update_layout(
                    title='💰 Top 10 Interests: Revenue vs Budget',
                    template='plotly_dark',
                    xaxis_title='Interest Category',
                    yaxis_title='Amount ($)',
                    yaxis=dict(tickformat='$,.0f'),
                    xaxis_tickangle=-45,
                    barmode='group',
                    height=350,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )

                st.plotly_chart(fig_interest, use_container_width=True)
            else:
                st.info("Interest data not available.")

    if 'Objective' in filtered_df.columns:
        st.markdown('<h2 class="sub-header" style="color: #ffd166;">🎯 Campaign Objective Performance</h2>', unsafe_allow_html=True)

        obj_col1, obj_col2 = st.columns(2)

        with obj_col1:
            objective_performance = filtered_df.groupby('Objective').agg({
                'Revenue': 'sum',
                'Budget_Spent': 'sum',
                'Profit': 'sum',
                'ROI': 'mean'
            }).reset_index().sort_values('ROI', ascending=False)

            fig_objective = go.Figure()

            fig_objective.add_trace(go.Bar(
                x=objective_performance['Objective'],
                y=objective_performance['ROI'],
                name='ROI',
                marker_color=objective_performance['ROI'].apply(
                    lambda x: '#00ff88' if x > 1 else ('#ffd166' if x > 0.5 else '#ff6b6b')
                ),
                text=objective_performance['ROI'].apply(lambda x: f"{x:.2f}x"),
                textposition='outside',
                hovertemplate="<b>%{x}</b><br>ROI: %{y:.2f}x<br>Revenue: $%{customdata[0]:,.0f}<br>Budget: $%{customdata[1]:,.0f}<br>Profit: $%{customdata[2]:,.0f}",
                customdata=objective_performance[['Revenue', 'Budget_Spent', 'Profit']].values
            ))

            fig_objective.update_layout(
                title='💰 ROI by Campaign Objective',
                template='plotly_dark',
                xaxis_title='Campaign Objective',
                yaxis_title='ROI (x)',
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=False,
                xaxis=dict(tickangle=-45 if len(objective_performance) > 3 else 0)
            )

            fig_objective.add_shape(
                type="line",
                x0=-0.5,
                x1=len(objective_performance)-0.5,
                y0=1,
                y1=1,
                line=dict(color="#ffffff", width=2, dash="dash"),
            )

            fig_objective.add_annotation(
                x=len(objective_performance)-1,
                y=1.1,
                text="Break-even (ROI = 1x)",
                showarrow=False,
                font=dict(color="#ffffff", size=10),
                bgcolor="rgba(0,0,0,0.5)",
                bordercolor="#ffffff",
                borderwidth=1,
                borderpad=4
            )

            st.plotly_chart(fig_objective, use_container_width=True)

        with obj_col2:
            fig_objective_scatter = px.scatter(
                objective_performance,
                x='Budget_Spent',
                y='Revenue',
                size='Profit',
                color='Objective',
                hover_name='Objective',
                title='📊 Objective: Budget vs Revenue',
                labels={
                    'Budget_Spent': 'Budget Spent ($)',
                    'Revenue': 'Revenue ($)',
                    'Profit': 'Profit (size)'
                },
                size_max=50,
                color_discrete_sequence=px.colors.qualitative.Bold
            )

            fig_objective_scatter.update_layout(
                template='plotly_dark',
                height=400,
                xaxis=dict(tickformat='$,.0f'),
                yaxis=dict(tickformat='$,.0f'),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )

            st.plotly_chart(fig_objective_scatter, use_container_width=True)

    st.markdown('<h2 class="sub-header">💡 Key Insights & Recommendations</h2>', unsafe_allow_html=True)

    insight_col1, insight_col2, insight_col3 = st.columns(3)

    with insight_col1:
        total_investment = filtered_df['Budget_Spent'].sum()
        total_return = filtered_df['Revenue'].sum()
        overall_roi = (total_return - total_investment) / total_investment if total_investment > 0 else 0
        best_platform = filtered_df.groupby('Platform_Name')['ROI'].mean().idxmax()
        best_platform_roi = filtered_df.groupby('Platform_Name')['ROI'].mean().max()

        st.markdown(f"""
        <div class="insight-box">
            <h4>💰 Financial Summary</h4>
            <p><strong>Total Investment:</strong> {format_large_number(total_investment)}</p>
            <p><strong>Total Return:</strong> {format_large_number(total_return)}</p>
            <p><strong>Overall ROI:</strong> {overall_roi:.2f}x</p>
            <p><strong>Best Platform:</strong> {best_platform} ({best_platform_roi:.2f}x ROI)</p>
            <p><strong>Recommendation:</strong> Increase budget for {best_platform} by 20%</p>
        </div>
        """, unsafe_allow_html=True)

    with insight_col2:
        best_location = filtered_df.groupby('Location')['ROI'].mean().idxmax() if 'Location' in filtered_df.columns else "N/A"
        best_location_roi = filtered_df.groupby('Location')['ROI'].mean().max() if 'Location' in filtered_df.columns else 0
        best_age_group = filtered_df.groupby('Age_Group')['ROI'].mean().idxmax() if 'Age_Group' in filtered_df.columns else "N/A"
        best_age_roi = filtered_df.groupby('Age_Group')['ROI'].mean().max() if 'Age_Group' in filtered_df.columns else 0
        best_gender = filtered_df.groupby('Gender')['ROI'].mean().idxmax() if 'Gender' in filtered_df.columns else "N/A"
        best_gender_roi = filtered_df.groupby('Gender')['ROI'].mean().max() if 'Gender' in filtered_df.columns else 0

        st.markdown(f"""
        <div class="recommendation-box">
            <h4>👥 Top Demographic Segments</h4>
            <p><strong>Best Location:</strong> {best_location} ({best_location_roi:.2f}x ROI)</p>
            <p><strong>Best Age Group:</strong> {best_age_group} ({best_age_roi:.2f}x ROI)</p>
            <p><strong>Best Gender:</strong> {best_gender} ({best_gender_roi:.2f}x ROI)</p>
            <p><strong>Strategy:</strong> Focus on {best_location}, target {best_age_group} {best_gender}</p>
        </div>
        """, unsafe_allow_html=True)

    with insight_col3:
        highest_cpa_platform = filtered_df.groupby('Platform_Name')['CPA'].mean().idxmax()
        highest_cpa = filtered_df.groupby('Platform_Name')['CPA'].mean().max()
        highest_cpc_platform = filtered_df.groupby('Platform_Name')['CPC'].mean().idxmax() if 'CPC' in filtered_df.columns else "N/A"
        highest_cpc = filtered_df.groupby('Platform_Name')['CPC'].mean().max() if 'CPC' in filtered_df.columns else 0
        lowest_ctr_platform = filtered_df.groupby('Platform_Name')['CTR'].mean().idxmin() if 'CTR' in filtered_df.columns else "N/A"
        lowest_ctr = filtered_df.groupby('Platform_Name')['CTR'].mean().min() if 'CTR' in filtered_df.columns else 0

        st.markdown(f"""
        <div class="warning-box">
            <h4>⚠️ Areas Needing Attention</h4>
            <p><strong>Highest CPA:</strong> {highest_cpa_platform} (${highest_cpa:.2f})</p>
            <p><strong>Highest CPC:</strong> {highest_cpc_platform} (${highest_cpc:.2f})</p>
            <p><strong>Lowest CTR:</strong> {lowest_ctr_platform} ({lowest_ctr:.2f}%)</p>
            <p><strong>Action:</strong> Review and optimize {highest_cpa_platform} campaigns</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<h2 class="sub-header">📋 Campaign Details</h2>', unsafe_allow_html=True)

    display_columns = [
        'Campaign_Name', 'Platform_Name', 'Start_Date', 'interest', 'Age_Group',
        'Gender', 'Objective', 'Language', 'Company_Name', 'status',
        'Revenue', 'Profit', 'ROI', 'CTR', 'CPA', 'Conversion_Rate',
        'Budget_Spent', 'Total_Budget', 'Impressions', 'Clicks', 'Conversions',
        'Location'
    ]

    available_columns = [col for col in display_columns if col in filtered_df.columns]
    display_df = filtered_df[available_columns].sort_values('Revenue', ascending=False)

    if 'Start_Date' in display_df.columns:
        display_df['Start_Date'] = display_df['Start_Date'].dt.strftime('%Y-%m-%d')

    format_mapping = {
        'Revenue': lambda x: format_large_number(x),
        'Profit': lambda x: format_large_number(x),
        'Budget_Spent': lambda x: format_large_number(x),
        'Total_Budget': lambda x: format_large_number(x),
        'ROI': lambda x: f"{x:.2f}x",
        'CTR': lambda x: f"{x:.2f}%",
        'CPA': lambda x: f"${x:.2f}",
        'Conversion_Rate': lambda x: f"{x:.1f}%",
        'Impressions': lambda x: format_large_number_no_symbol(x),
        'Clicks': lambda x: format_large_number_no_symbol(x),
        'Conversions': lambda x: format_large_number_no_symbol(x)
    }

    for col, formatter in format_mapping.items():
        if col in display_df.columns:
            try:
                display_df[col] = display_df[col].apply(formatter)
            except:
                pass

    st.dataframe(display_df, use_container_width=True, height=500)

    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Full Filtered Data",
        data=csv,
        file_name=f"campaign_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        type="primary"
    )

    st.markdown("---")
    st.markdown("## 📊 Summary Statistics")

    sum_col1, sum_col2, sum_col3, sum_col4, sum_col5 = st.columns(5)

    with sum_col1:
        total_campaigns = len(filtered_df)
        st.metric("Total Campaigns", format_large_number_no_symbol(total_campaigns))

    with sum_col2:
        success_rate = (filtered_df['ROI'] > 1).mean() * 100
        st.metric("Success Rate (ROI > 1)", f"{success_rate:.1f}%")

    with sum_col3:
        if 'CPC' in filtered_df.columns:
            avg_cpc = filtered_df['CPC'].mean()
            st.metric("Avg CPC", f"${avg_cpc:.2f}")
        else:
            avg_cpa = filtered_df['CPA'].mean()
            st.metric("Avg CPA", f"${avg_cpa:.2f}")

    with sum_col4:
        Total_Profit = filtered_df['Profit'].sum()
        st.metric("Total Profit", format_large_number(Total_Profit))

    with sum_col5:
        if 'Total_Budget' in filtered_df.columns:
            total_budget = filtered_df['Total_Budget'].sum()
            st.metric("Total Budget", format_large_number(total_budget))
        else:
            avg_roi = filtered_df['ROI'].mean()
            st.metric("Average ROI", f"{avg_roi:.2f}x")

    st.markdown("---")
    data_source_info = "Default Data File" if st.session_state.data_source == 'default' else "Uploaded File"
    st.markdown(f"""
    <div style="text-align: center; color: #888; padding: 20px; font-size: 0.9em;">
        <p>📊 <strong>Dashboard Analytics Summary:</strong></p>
        <p>• Data Source: {data_source_info}</p>
        <p>• Showing {len(filtered_df)} campaigns (Filtered from {len(df)} total)</p>
        <p>• Date Range: {start_date} to {end_date}</p>
        <p>• ROI Threshold: ≥ {min_roi}x | Budget Range: ${min_budget}-${max_budget}</p>
        <p>• Total Revenue Generated: {format_large_number(filtered_df['Revenue'].sum())}</p>
        <p>• Total Profit: {format_large_number(filtered_df['Profit'].sum())}</p>
        <p>• Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>💡 <strong>Tip:</strong> Use filters to analyze specific segments. Click 'Generate Report' for detailed analysis.</p>
    </div>
    """, unsafe_allow_html=True)

# Report Page
elif st.session_state.page == 'report':
    st.markdown('<div class="report-header">📋 Comprehensive Campaign Performance Report</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="report-section">
            <h4>📅 Report Period</h4>
            <p><strong>Start Date:</strong> {start_date}</p>
            <p><strong>End Date:</strong> {end_date}</p>
            <p><strong>Duration:</strong> {(end_date - start_date).days} days</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="report-section">
            <h4>📊 Campaign Overview</h4>
            <p><strong>Total Campaigns:</strong> {format_large_number_no_symbol(len(filtered_df))}</p>
            <p><strong>Success Rate:</strong> {(filtered_df['ROI'] > 1).mean() * 100:.1f}%</p>
            <p><strong>Active Platforms:</strong> {filtered_df['Platform_Name'].nunique()}</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        total_revenue = filtered_df['Revenue'].sum()
        total_profit = filtered_df['Profit'].sum()
        st.markdown(f"""
        <div class="report-section">
            <h4>💰 Financial Summary</h4>
            <div class="report-metric">{format_large_number(total_revenue)}</div>
            <p><strong>Total Revenue</strong></p>
            <div class="report-metric" style="color: #00ff88;">{format_large_number(total_profit)}</div>
            <p><strong>Total Profit</strong></p>
        </div>
        """, unsafe_allow_html=True)

    report_content = generate_comprehensive_report(filtered_df)

    st.markdown("---")

    st.markdown("## 📈 Executive Summary")
    exec_col1, exec_col2, exec_col3, exec_col4 = st.columns(4)

    with exec_col1:
        avg_roi = filtered_df['ROI'].mean()
        st.metric("Average ROI", f"{avg_roi:.2f}x")

    with exec_col2:
        avg_cpa = filtered_df['CPA'].mean()
        st.metric("Average CPA", f"${avg_cpa:.2f}")

    with exec_col3:
        total_conversions = filtered_df['Conversions'].sum()
        st.metric("Total Conversions", format_large_number_no_symbol(total_conversions))

    with exec_col4:
        if 'Budget_Utilization' in filtered_df.columns:
            budget_utilization = filtered_df['Budget_Utilization'].mean()
            st.metric("Budget Utilization", f"{budget_utilization:.1f}%")
        else:
            total_spent = filtered_df['Budget_Spent'].sum()
            st.metric("Total Spent", format_large_number(total_spent))

    st.markdown("## 🏆 Platform Performance Analysis")

    platform_col1, platform_col2 = st.columns(2)

    with platform_col1:
        top_platforms = filtered_df.groupby('Platform_Name')['ROI'].mean().sort_values(ascending=False).head(3)

        st.markdown("### 🥇 Top Performing Platforms")
        for platform, roi in top_platforms.items():
            revenue = filtered_df[filtered_df['Platform_Name'] == platform]['Revenue'].sum()
            budget = filtered_df[filtered_df['Platform_Name'] == platform]['Budget_Spent'].sum()

            if roi > 1:
                badge_class = "badge-success"
                status = "Excellent"
            elif roi > 0.5:
                badge_class = "badge-warning"
                status = "Good"
            else:
                badge_class = "badge-danger"
                status = "Needs Improvement"

            st.markdown(f"""
            <div style="background: rgba(30, 30, 46, 0.7); padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 5px solid #00d4ff;">
                <h4>{platform}</h4>
                <span class="badge {badge_class}">{status}</span>
                <p><strong>ROI:</strong> {roi:.2f}x</p>
                <p><strong>Revenue:</strong> {format_large_number(revenue)}</p>
                <p><strong>Budget:</strong> {format_large_number(budget)}</p>
            </div>
            """, unsafe_allow_html=True)

    with platform_col2:
        platform_metrics_data = filtered_df.groupby('Platform_Name').agg({
            'CPA': 'mean',
            'ROI': 'mean'
        }).round(2)

        if 'CPC' in filtered_df.columns:
            platform_metrics_data['CPC'] = filtered_df.groupby('Platform_Name')['CPC'].mean().round(2)
        if 'CTR' in filtered_df.columns:
            platform_metrics_data['CTR'] = filtered_df.groupby('Platform_Name')['CTR'].mean().round(2)
        if 'Conversion_Rate' in filtered_df.columns:
            platform_metrics_data['Conversion_Rate'] = filtered_df.groupby('Platform_Name')['Conversion_Rate'].mean().round(1)

        st.markdown("### 📊 Platform Efficiency Metrics")
        st.dataframe(platform_metrics_data, use_container_width=True)

    st.markdown("## 👥 Demographic Performance Insights")

    demo_tab1, demo_tab2, demo_tab3 = st.tabs(["📍 Location", "👤 Age & Gender", "🎯 Interests"])

    with demo_tab1:
        if 'Location' in filtered_df.columns:
            location_performance = filtered_df.groupby('Location').agg({
                'Revenue': 'sum',
                'ROI': 'mean',
                'Budget_Spent': 'sum'
            }).sort_values('ROI', ascending=False)

            st.markdown("### 🌍 Top Performing Locations")
            col1, col2, col3 = st.columns(3)

            top_locations = location_performance.head(3)
            for idx, (col, (location, data)) in enumerate(zip([col1, col2, col3], top_locations.iterrows())):
                with col:
                    st.markdown(f"""
                    <div style="text-align: center; padding: 20px; background: rgba(30, 30, 46, 0.7); border-radius: 10px;">
                        <h3 style="color: #00d4ff;">#{idx + 1}</h3>
                        <h4>{location}</h4>
                        <div style="font-size: 1.5rem; font-weight: bold; color: #00ff88;">{data['ROI']:.2f}x</div>
                        <p>ROI</p>
                        <p><strong>Revenue:</strong> {format_large_number(data['Revenue'])}</p>
                        <p><strong>Budget:</strong> {format_large_number(data['Budget_Spent'])}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Location data not available.")

    with demo_tab2:
        if 'Age_Group' in filtered_df.columns and 'Gender' in filtered_df.columns:
            age_gender_performance = filtered_df.groupby(['Age_Group', 'Gender']).agg({
                'ROI': 'mean',
                'Revenue': 'sum',
                'Conversion_Rate': 'mean' if 'Conversion_Rate' in filtered_df.columns else 'ROI'
            }).reset_index()

            fig_age_gender = px.bar(
                age_gender_performance,
                x='Age_Group',
                y='ROI',
                color='Gender',
                barmode='group',
                title='📊 ROI by Age Group and Gender',
                color_discrete_sequence=['#00d4ff', '#ff6b6b']
            )

            fig_age_gender.update_layout(template='plotly_dark', height=400)
            st.plotly_chart(fig_age_gender, use_container_width=True)
        else:
            st.info("Age & Gender data not available.")

    with demo_tab3:
        if 'interest' in filtered_df.columns:
            interest_performance = filtered_df.groupby('interest').agg({
                'Revenue': 'sum',
                'ROI': 'mean',
                'Conversions': 'sum'
            }).sort_values('Revenue', ascending=False).head(10)

            fig_interest = px.pie(
                interest_performance,
                values='Revenue',
                names=interest_performance.index,
                title='💰 Revenue Distribution by Interest',
                hole=0.4
            )

            fig_interest.update_layout(template='plotly_dark', height=400)
            st.plotly_chart(fig_interest, use_container_width=True)
        else:
            st.info("Interest data not available.")

    st.markdown("## 💡 Strategic Recommendations")

    rec_col1, rec_col2 = st.columns(2)

    with rec_col1:
        st.markdown("""
        <div class="insight-box">
            <h4>🚀 Immediate Actions</h4>
            <ol>
                <li><strong>Optimize Budget Allocation:</strong> Shift 30% of budget from low-performing platforms to top performers</li>
                <li><strong>Campaign Optimization:</strong> Improve ad creatives on platforms with low CTR</li>
                <li><strong>Audience Targeting:</strong> Refine targeting based on top demographic segments</li>
                <li><strong>Cost Control:</strong> Set CPA/CPC caps on high-cost platforms</li>
                <li><strong>A/B Testing:</strong> Implement systematic testing of ad variations</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    with rec_col2:
        st.markdown("""
        <div class="recommendation-box">
            <h4>📈 Long-term Strategies</h4>
            <ol>
                <li><strong>Platform Diversification:</strong> Test 2 new advertising platforms quarterly</li>
                <li><strong>Data Integration:</strong> Implement advanced analytics and attribution modeling</li>
                <li><strong>Automation:</strong> Deploy AI-driven bid optimization tools</li>
                <li><strong>Content Strategy:</strong> Develop platform-specific content calendars</li>
                <li><strong>Team Training:</strong> Monthly workshops on latest advertising trends</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("## ⚠️ Risk Assessment & Areas for Improvement")

    risk_metrics = filtered_df.groupby('Platform_Name').agg({
        'CPA': 'mean',
        'ROI': 'mean'
    })

    if 'CPC' in filtered_df.columns:
        risk_metrics['CPC'] = filtered_df.groupby('Platform_Name')['CPC'].mean()
    if 'CTR' in filtered_df.columns:
        risk_metrics['CTR'] = filtered_df.groupby('Platform_Name')['CTR'].mean()

    high_cpa = risk_metrics['CPA'].idxmax()
    high_cpa_value = risk_metrics['CPA'].max()

    if 'CTR' in risk_metrics.columns:
        low_ctr = risk_metrics['CTR'].idxmin()
        low_ctr_value = risk_metrics['CTR'].min()
    else:
        low_ctr = "N/A"
        low_ctr_value = 0

    low_roi = risk_metrics['ROI'].idxmin()
    low_roi_value = risk_metrics['ROI'].min()

    risk_col1, risk_col2, risk_col3 = st.columns(3)

    with risk_col1:
        st.markdown(f"""
        <div class="warning-box">
            <h4>💸 High Cost Platform</h4>
            <h3>{high_cpa}</h3>
            <div style="font-size: 1.5rem; color: #ff6b6b;">${high_cpa_value:.2f}</div>
            <p>Average CPA</p>
            <p><strong>Action:</strong> Review targeting and adjust bids</p>
        </div>
        """, unsafe_allow_html=True)

    with risk_col2:
        if low_ctr != "N/A":
            st.markdown(f"""
            <div class="warning-box">
                <h4>📉 Low Engagement</h4>
                <h3>{low_ctr}</h3>
                <div style="font-size: 1.5rem; color: #ffd166;">{low_ctr_value:.2f}%</div>
                <p>Click-through Rate</p>
                <p><strong>Action:</strong> Test new ad creatives</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="warning-box">
                <h4>📉 Low Performance</h4>
                <h3>{low_roi}</h3>
                <div style="font-size: 1.5rem; color: #ff6b6b;">{low_roi_value:.2f}x</div>
                <p>Return on Investment</p>
                <p><strong>Action:</strong> Consider reducing budget</p>
            </div>
            """, unsafe_allow_html=True)

    with risk_col3:
        st.markdown(f"""
        <div class="warning-box">
            <h4>📊 Low ROI Platform</h4>
            <h3>{low_roi}</h3>
            <div style="font-size: 1.5rem; color: #ff6b6b;">{low_roi_value:.2f}x</div>
            <p>Return on Investment</p>
            <p><strong>Action:</strong> Consider reducing budget</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 📥 Download Complete Report")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        report_txt = generate_comprehensive_report(filtered_df)

        st.download_button(
            label="📄 Download Full Report (TXT)",
            data=report_txt,
            file_name=f"campaign_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )

        html_report = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Campaign Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
                .header {{ background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); color: white; padding: 30px; border-radius: 10px; }}
                .section {{ background: white; padding: 20px; margin: 20px 0; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .metric {{ font-size: 2rem; font-weight: bold; color: #00d4ff; }}
                .badge {{ display: inline-block; padding: 5px 15px; border-radius: 20px; color: white; margin: 5px; }}
                .success {{ background: #00b894; }}
                .warning {{ background: #fdcb6e; }}
                .danger {{ background: #d63031; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background: #1a1a2e; color: white; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 Campaign Performance Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Period: {start_date} to {end_date}</p>
            </div>
            
            <div class="section">
                <h2>📈 Executive Summary</h2>
                <p><strong>Total Campaigns:</strong> {len(filtered_df):,}</p>
                <p><strong>Total Revenue:</strong> {format_large_number(filtered_df['Revenue'].sum())}</p>
                <p><strong>Total Profit:</strong> {format_large_number(filtered_df['Profit'].sum())}</p>
                <p><strong>Average ROI:</strong> {filtered_df['ROI'].mean():.2f}x</p>
                <p><strong>Success Rate:</strong> {(filtered_df['ROI'] > 1).mean() * 100:.1f}%</p>
            </div>
            
            <div class="section">
                <h2>🏆 Top Performers</h2>
                <p><strong>Best Platform:</strong> {filtered_df.groupby('Platform_Name')['ROI'].mean().idxmax()}</p>
                <p><strong>Best Location:</strong> {filtered_df.groupby('Location')['ROI'].mean().idxmax() if 'Location' in filtered_df.columns else 'N/A'}</p>
                <p><strong>Best Age Group:</strong> {filtered_df.groupby('Age_Group')['ROI'].mean().idxmax() if 'Age_Group' in filtered_df.columns else 'N/A'}</p>
            </div>
            
            <div class="section">
                <h2>💡 Recommendations</h2>
                <ol>
                    <li>Increase budget allocation for top-performing platforms by 20%</li>
                    <li>Optimize ad creatives for platforms with low CTR</li>
                    <li>Expand campaigns in high-performing geographic locations</li>
                    <li>Implement A/B testing for campaign optimization</li>
                    <li>Review and adjust bidding strategies for high CPA platforms</li>
                </ol>
            </div>
        </body>
        </html>
        """

        st.download_button(
            label="🌐 Download HTML Report",
            data=html_report,
            file_name=f"campaign_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html",
            use_container_width=True
        )

    if st.button("⬅️ Back to Dashboard", use_container_width=True):
        st.session_state.page = 'dashboard'
        st.rerun()