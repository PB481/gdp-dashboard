import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import io
import base64
import random # For simulating security complexity
import numpy as np # For numerical operations

st.set_page_config(layout="wide", page_title="Fund Analytics Dashboard")

# --- Helper Functions ---
def load_excel_data(uploaded_file, sheet_name=None):
    """Loads data from an uploaded Excel file."""
    if uploaded_file is not None:
        try:
            if sheet_name:
                df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
            else:
                df = pd.read_excel(uploaded_file)
            return df
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None
    return None

def generate_html_report(performance_plot_html, transactions_plot_html, clustering_plot_html, insights_html):
    """Generates a simple HTML report."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fund Performance Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #333; }}
            .section {{ margin-bottom: 40px; border: 1px solid #eee; padding: 20px; border-radius: 8px; }}
        </style>
    </head>
    <body>
        <h1>Fund Performance and Portfolio Analysis Report</h1>

        <div class="section">
            <h2>1. Fund Performance</h2>
            {performance_plot_html}
        </div>

        <div class="section">
            <h2>2. Security Transactions Analysis</h2>
            {transactions_plot_html}
        </div>

        <div class="section">
            <h2>3. Fund Commonalities (Clustering)</h2>
            {clustering_plot_html}
        </div>

        <div class="section">
            <h2>4. Key Insights and Analysis</h2>
            {insights_html}
        </div>

    </body>
    </html>
    """
    return html_content

# --- Main Streamlit App ---
st.title("💰 Fund Performance and Portfolio Analytics")

# --- File Upload Section ---
st.sidebar.header("Upload Data")
st.sidebar.markdown("Please upload your Excel files for analysis.")

performance_file = st.sidebar.file_uploader("Upload Fund Performance Data (Sheet: 'Performance')", type=["xlsx"])
portfolio_file = st.sidebar.file_uploader("Upload Portfolio Holdings Data (Sheet: 'Portfolio')", type=["xlsx"])
transactions_file = st.sidebar.file_uploader("Upload Transaction Data (Sheet: 'Transactions')", type=["xlsx"])

# --- Data Loading and Preprocessing ---
fund_performance_df = None
portfolio_df = None
transactions_df = None

if performance_file:
    fund_performance_df = load_excel_data(performance_file, sheet_name='Performance')
    if fund_performance_df is not None:
        st.sidebar.success("Fund Performance Data Loaded!")
        # Expected columns: ID, Fund Name, Price, Date
        fund_performance_df['Date'] = pd.to_datetime(fund_performance_df['Date'])
        fund_performance_df = fund_performance_df.sort_values(by=['Fund Name', 'Date'])

if portfolio_file:
    portfolio_df = load_excel_data(portfolio_file, sheet_name='Portfolio')
    if portfolio_df is not None:
        st.sidebar.success("Portfolio Holdings Data Loaded!")
        # Expected columns: Fund, Fund IS, Security ID and Name, Industry, Market Value, Shares held

if transactions_file:
    transactions_df = load_excel_data(transactions_file, sheet_name='Transactions')
    if transactions_df is not None:
        st.sidebar.success("Transaction Data Loaded!")
        # Expected columns: Fund, Fund ID, Security ID, Shares, Price, Base Market Value, Purchase or Sale

# --- Analysis and Visualization Sections ---

# 1. Fund Performance Graph
st.header("1. Fund Performance Analysis")
if fund_performance_df is not None and not fund_performance_df.empty:
    selected_funds_performance = st.multiselect(
        "Select Funds for Performance Chart",
        options=fund_performance_df['Fund Name'].unique(),
        default=fund_performance_df['Fund Name'].unique()[:min(5, len(fund_performance_df['Fund Name'].unique()))]
    )
    if selected_funds_performance:
        filtered_performance_df = fund_performance_df[fund_performance_df['Fund Name'].isin(selected_funds_performance)]
        fig_performance = px.line(
            filtered_performance_df,
            x="Date",
            y="Price",
            color="Fund Name",
            title="Fund Performance Over Time"
        )
        st.plotly_chart(fig_performance, use_container_width=True)
        performance_plot_html = fig_performance.to_html(full_html=False, include_plotlyjs='cdn')
    else:
        st.info("Please select at least one fund to view performance.")
        performance_plot_html = "<i>No funds selected for performance chart.</i>"
else:
    st.info("Please upload Fund Performance Data to view this section.")
    performance_plot_html = "<i>Fund Performance Data not available.</i>"

# 2. Security Transactions Analysis
st.header("2. Security Transactions Analysis")
if transactions_df is not None and not transactions_df.empty:
    st.subheader("Transactions by Share Volume")
    transaction_volume = transactions_df.groupby(['Fund', 'Security ID']).agg(
        Total_Shares=('Shares', 'sum')
    ).reset_index().sort_values(by='Total_Shares', ascending=False)

    fig_volume = px.bar(
        transaction_volume.head(20), # Show top 20 for readability
        x="Security ID",
        y="Total_Shares",
        color="Fund",
        title="Top 20 Security Transactions by Share Volume (Overall)"
    )
    st.plotly_chart(fig_volume, use_container_width=True)

    st.subheader("Number of Transactions by Fund")
    transactions_by_fund = transactions_df.groupby('Fund').size().reset_index(name='Number of Transactions')
    fig_num_transactions = px.bar(
        transactions_by_fund,
        x="Fund",
        y="Number of Transactions",
        title="Number of Transactions by Fund"
    )
    st.plotly_chart(fig_num_transactions, use_container_width=True)
    transactions_plot_html = fig_volume.to_html(full_html=False, include_plotlyjs='cdn') + fig_num_transactions.to_html(full_html=False, include_plotlyjs='cdn')
else:
    st.info("Please upload Transaction Data to view this section.")
    transactions_plot_html = "<i>Transaction Data not available.</i>"

# 3. Clustering Graph and Fund Commonalities
st.header("3. Fund Commonalities using Clustering")
if portfolio_df is not None and not portfolio_df.empty:
    st.markdown("This section analyzes the common relationships of funds based on their underlying portfolio holdings, industry exposure, and simulated transaction volume.")

    # --- Feature Engineering for Clustering ---
    st.subheader("Clustering Parameters and Feature Engineering")

    # Group portfolio by Fund to get an aggregated view
    fund_portfolio_agg = portfolio_df.groupby('Fund').agg(
        Total_Market_Value=('Market Value', 'sum'),
        Unique_Securities=('Security ID', 'nunique')
    ).reset_index()

    # Get industry exposure for each fund
    # Pivot table to get industry exposure for each fund
    industry_exposure = portfolio_df.pivot_table(
        index='Fund',
        columns='Industry',
        values='Market Value',
        aggfunc='sum',
        fill_value=0
    ).fillna(0) # Fill NaNs after pivot

    # Merge aggregated data with industry exposure
    clustering_df = fund_portfolio_agg.set_index('Fund').join(industry_exposure).reset_index()

    # Add simulated transaction volume for clustering
    if transactions_df is not None and not transactions_df.empty:
        fund_transaction_volume = transactions_df.groupby('Fund').agg(
            Total_Shares_Traded=('Shares', 'sum')
        ).reset_index()
        clustering_df = pd.merge(clustering_df, fund_transaction_volume, on='Fund', how='left').fillna(0)
    else:
        st.warning("Transaction data not available for transaction volume in clustering. Simulating zeros.")
        clustering_df['Total_Shares_Traded'] = 0

    st.write("Features used for clustering:")
    st.write(clustering_df.head())

    # User input for number of clusters (shapes)
    max_num_shapes = st.slider("Maximum number of fund shapes (clusters)", min_value=2, max_value=10, value=5)

    # Prepare data for clustering
    features = clustering_df.drop(columns=['Fund'])
    # Handle non-numeric columns if any before scaling
    numeric_features = features.select_dtypes(include=np.number)

    if not numeric_features.empty:
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(numeric_features)

        # Determine optimal number of clusters using Elbow Method (simplified for app)
        # In a real app, you might run elbow method and display plot for user to decide
        # For simplicity, we'll just use the user-selected max_num_shapes
        n_clusters = max_num_shapes # Or you could try to determine programmatically

        if n_clusters > len(scaled_features):
            st.warning(f"Number of clusters ({n_clusters}) is greater than the number of funds ({len(scaled_features)}). Adjusting clusters to number of funds.")
            n_clusters = len(scaled_features)
            if n_clusters < 2:
                st.warning("Not enough funds for clustering.")
                clustering_plot_html = "<i>Not enough funds for clustering.</i>"
                st.stop() # Stop execution if not enough data


        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # n_init for robustness
        clustering_df['Cluster'] = kmeans.fit_predict(scaled_features)

        st.subheader("Fund Clusters (Shapes)")
        fig_cluster = px.scatter(
            clustering_df,
            x="Total_Market_Value", # Example X-axis, can be changed
            y="Unique_Securities",  # Example Y-axis, can be changed
            color="Cluster",
            hover_data=['Fund'] + list(numeric_features.columns),
            title=f"Fund Commonalities (Clustering - {n_clusters} Shapes)"
        )
        st.plotly_chart(fig_cluster, use_container_width=True)
        clustering_plot_html = fig_cluster.to_html(full_html=False, include_plotlyjs='cdn')

        # Determine most common shape
        most_common_shape = clustering_df['Cluster'].mode()[0]
        st.info(f"The most common fund shape (cluster) is Cluster {most_common_shape}.")

        st.subheader("Funds per Cluster")
        st.dataframe(clustering_df[['Fund', 'Cluster']].sort_values(by='Cluster'))

        # Analyze cluster characteristics (optional, but very useful)
        st.subheader("Cluster Characteristics")
        cluster_summary = clustering_df.groupby('Cluster')[numeric_features.columns].mean()
        st.dataframe(cluster_summary)
        st.markdown("Each row above represents a 'shape' or cluster, showing the average characteristics of funds within that cluster.")

    else:
        st.warning("No numeric features found for clustering. Please check your portfolio data.")
        clustering_plot_html = "<i>Not enough numeric features for clustering.</i>"
else:
    st.info("Please upload Portfolio Holdings Data to view this section.")
    clustering_plot_html = "<i>Portfolio Holdings Data not available.</i>"

# Highest and Lowest Volume of Assets Traded
st.header("Highest and Lowest Volume of Assets Traded")
if transactions_df is not None and not transactions_df.empty:
    transactions_df['Total_Value'] = transactions_df['Shares'] * transactions_df['Price']
    top_assets_traded = transactions_df.groupby('Security ID')['Total_Value'].sum().nlargest(10).reset_index()
    bottom_assets_traded = transactions_df.groupby('Security ID')['Total_Value'].sum().nsmallest(10).reset_index()

    st.subheader("Top 10 Assets by Total Traded Value")
    st.dataframe(top_assets_traded)

    st.subheader("Bottom 10 Assets by Total Traded Value")
    st.dataframe(bottom_assets_traded)
    insights_html_traded = f"""
    <h3>Highest and Lowest Volume of Assets Traded</h3>
    <p><b>Top 10 Assets by Total Traded Value:</b></p>
    {top_assets_traded.to_html(index=False)}
    <p><b>Bottom 10 Assets by Total Traded Value:</b></p>
    {bottom_assets_traded.to_html(index=False)}
    """
else:
    st.info("Please upload Transaction Data to view this section.")
    insights_html_traded = "<i>Transaction Data not available for trade volume analysis.</i>"


# Add Complexity to Each Security Type
st.header("Security Complexity (User Defined)")
st.markdown("Here, you can define a 'complexity score' for different security types.")

# Example: User can define complexity for certain industries or security IDs
if portfolio_df is not None and not portfolio_df.empty:
    unique_industries = portfolio_df['Industry'].unique().tolist()
    st.subheader("Define Complexity by Industry")
    complexity_scores = {}
    for industry in unique_industries:
        complexity_scores[industry] = st.slider(f"Complexity for '{industry}'", 0, 10, 5)

    portfolio_df['Security_Complexity'] = portfolio_df['Industry'].map(complexity_scores).fillna(0)

    st.subheader("Portfolio with User-Defined Security Complexity")
    st.dataframe(portfolio_df[['Fund', 'Security ID', 'Industry', 'Security_Complexity']].head(10)) # Show a sample

    # Example: You can use this complexity score in further analysis or visualization
    fig_complexity = px.scatter(
        portfolio_df,
        x="Market Value",
        y="Shares held",
        color="Security_Complexity",
        hover_data=['Fund', 'Security ID', 'Industry'],
        title="Portfolio Holdings by Market Value vs. Shares Held (Color by Complexity)"
    )
    st.plotly_chart(fig_complexity, use_container_width=True)
    insights_html_complexity = f"""
    <h3>Security Complexity (User Defined)</h3>
    <p>Security complexity scores defined by the user for various industries.</p>
    <p><b>Complexity Scores:</b> {complexity_scores}</p>
    """
else:
    st.info("Please upload Portfolio Holdings Data to define security complexity.")
    insights_html_complexity = "<i>Portfolio Holdings Data not available for defining security complexity.</i>"

# --- HTML Report Generation ---
st.sidebar.header("Generate Report")
if st.sidebar.button("Generate HTML Report"):
    if performance_plot_html and transactions_plot_html and clustering_plot_html:
        insights_html = f"""
        <h3>Key Insights:</h3>
        {insights_html_traded}
        {insights_html_complexity}
        <p><i>Further insights can be added programmatically here.</i></p>
        """
        html_report = generate_html_report(performance_plot_html, transactions_plot_html, clustering_plot_html, insights_html)
        b64 = base64.b64encode(html_report.encode()).decode()
        href = f'<a href="data:text/html;base64,{b64}" download="fund_analytics_report.html">Download HTML Report</a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)
        st.sidebar.success("Report generated successfully!")
    else:
        st.sidebar.warning("Please upload all necessary data and generate plots before creating a report.")
