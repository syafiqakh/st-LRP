import streamlit as st
import requests
import polars as pl
import pandas as pd
import sqlite3
import plotly.express as px
import math
from requests.auth import HTTPBasicAuth

# Define the number of rows per page for pagination
ROWS_PER_PAGE = 500

# Configure the Streamlit page (this must be the first Streamlit command)
st.set_page_config(page_title="Flash Planner", page_icon="📊", layout="wide")

# Database Connection Setup
def create_connection():
    return sqlite3.connect("sqlite_db_name.db")  # Replace with your actual database file

@st.cache_data()
def load_build_plan_data():
    try:
        connection = create_connection()
        query = '''
            SELECT ITEM, DEVICE, SITEID, SDE, BUILDQTY, CALENDAR_MONTH, CALENDAR_QTR, UPSERT_TIME
            FROM build_plan
        '''
        data = pd.read_sql_query(query, connection)
        connection.close()
        return pl.DataFrame(data)
    except Exception as e:
        st.error(f"Error loading build plan data: {e}")
        return pl.DataFrame()

def load_build_plan_for_chart():
    """
    Load build plan data directly from SQLite for use in the stacked area chart.
    """
    try:
        connection = create_connection()
        query = '''
            SELECT CALENDAR_MONTH, NODE, FINANCE_PB_BY_SITE
            FROM build_plan
        '''
        data = pd.read_sql_query(query, connection)
        connection.close()
        return pl.DataFrame(data)  # Convert to Polars Data
    except Exception as e:
        st.error(f"Error loading build plan data for chart: {e}")
        return pl.DataFrame()

# Function to display the stacked area chart
def show_stacked_area_chart(data):
    required_columns = ["CALENDAR_MONTH", "NODE", "FINANCE_PB_BY_SITE"]

    # Check if the required columns are present
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        st.error(f"The data must contain the following columns: {required_columns}. Missing columns: {missing_columns}")
        st.write("Available columns in data:", list(data.columns))
        return

    # Create the stacked area chart
    data_pandas = data.to_pandas()
    grouped_data = (
        data_pandas.groupby(["CALENDAR_MONTH", "NODE"])["FINANCE_PB_BY_SITE"]
        .sum()
        .reset_index()
    )
    total_by_month = grouped_data.groupby("CALENDAR_MONTH")["FINANCE_PB_BY_SITE"].transform("sum")
    grouped_data["PERCENTAGE"] = (grouped_data["FINANCE_PB_BY_SITE"] / total_by_month) * 100

    fig = px.area(
        grouped_data,
        x="CALENDAR_MONTH",
        y="PERCENTAGE",
        color="NODE",
        title="SIP Technology (PB)",
        labels={
            "PERCENTAGE": "Percentage of Finance PB",
            "CALENDAR_MONTH": "Calendar Month",
            "NODE": "Node",
        },
        template="plotly_white",
    )

    fig.update_layout(
        xaxis_title="Calendar Month",
        yaxis_title="Percentage",
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(0, 101, 10)),
            ticktext=[f"{x}%" for x in range(0, 101, 10)],
        ),
        showlegend=True,
    )

    st.plotly_chart(fig, use_container_width=True)


def load_uph_goal_data():
    try:
        connection = create_connection()
        query = 'SELECT * FROM uph_goal'
        data = pd.read_sql_query(query, connection)
        connection.close()
        
        # Convert to Polars DataFrame
        df = pl.DataFrame(data)
        
        # Extract base recipe by removing the suffix from RECIPE_ID (e.g., P09824-1 -> P09824)
        df = df.with_columns(
            pl.col("RECIPE_ID").str.split('-').list.first().alias("BASE_RECIPE")
        )
        
        # Group recipes with the same BASE_RECIPE into a RECIPE_GROUP
        df = df.with_columns(
            pl.col("BASE_RECIPE").alias("RECIPE_GROUP")
        )
        
        # Convert to pandas for group-wise calculation
        df = df.to_pandas()

        # Group by RECIPE_GROUP and calculate the new UPH
        def calculate_new_uph(group):
            uph_values = group['UPH_GOAL']
            if len(uph_values) > 1:
                new_uph = 1 / (1 / uph_values.iloc[0] + 1 / uph_values.iloc[1])
            else:
                new_uph = uph_values.iloc[0]
            return pd.Series({"NEW_UPH": new_uph})

        new_uph_df = df.groupby("RECIPE_GROUP").apply(calculate_new_uph).reset_index()
        df = df.merge(new_uph_df, on="RECIPE_GROUP", how="left")
        
        # Calculate TOOL_DAILY_CAPACITY
        df['TOOL_DAILY_CAPACITY'] = df['NEW_UPH'] * 30 * 0.88  # 30 days with an efficiency factor

        # Calculate TOOL_MONTHLY_CAPACITY
        df['TOOL_MONTHLY_CAPACITY'] = df['TOOL_DAILY_CAPACITY'] * 30  # Multiply daily capacity by 30 days

        # Remove duplicates in RECIPE_GROUP, keeping the first entry
        df = df.drop_duplicates(subset="RECIPE_GROUP").reset_index(drop=True)

        # Convert back to Polars DataFrame
        df = pl.DataFrame(df)

        return df
    except Exception as e:
        st.error(f"Error loading UPH goal data: {e}")
        return pl.DataFrame()

def load_build_plan_sde():
    """
    Load raw build plan data directly from SQLite for SDE loading visualization.
    """
    try:
        connection = create_connection()
        query = '''
            SELECT SITEID, SDE, BUILDQTY, CALENDAR_QTR
            FROM build_plan
        '''
        data = pd.read_sql_query(query, connection)
        connection.close()
        return pl.DataFrame(data)  # Convert to Polars DataFrame
    except Exception as e:
        st.error(f"Error loading build plan data for SDE loading chart: {e}")
        return pl.DataFrame()

def show_sde_loading_chart(data):
    """
    Create and display a chart for SDE loading grouped by SITEID and CALENDAR_QTR.
    """
    required_columns = ["SITEID", "SDE", "BUILDQTY", "CALENDAR_QTR"]

    # Check for required columns
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        st.error(f"The data must contain the following columns: {required_columns}. Missing columns: {missing_columns}")
        st.write("Available columns in data:", list(data.columns))
        return

    # Group data for the chart
    data_pandas = data.to_pandas()
    grouped_data = (
        data_pandas.groupby(["CALENDAR_QTR", "SITEID"])["BUILDQTY"]
        .sum()
        .reset_index()
    )

    # Create a stacked bar chart for SDE loading
    fig = px.bar(
        grouped_data,
        x="CALENDAR_QTR",
        y="BUILDQTY",
        color="SITEID",
        title="Site SDE Loading (SDE)",
        labels={
            "BUILDQTY": "Build Quantity",
            "CALENDAR_QTR": "Calendar Quarter",
            "SITEID": "Site ID",
        },
        template="plotly_white",
    )

    fig.update_layout(
        xaxis_title="Calendar Quarter",
        yaxis_title="Build Quantity",
        barmode="stack",
        showlegend=True,
    )

    st.plotly_chart(fig, use_container_width=True)

# Sidebar configuration
st.sidebar.title("IE Capacity Planning Tool")
st.sidebar.write("Analytics | Compute | Upload user data")

# Filters for SITEID
site_id_options = ["C039", "C040", "C041"]
selected_site_ids = st.sidebar.multiselect("Site IDs", options=site_id_options, default=site_id_options[0])

# Main Category
main_category_options = ["CALENDAR_MONTH", "PRODUCT_ID"]
selected_main_category = st.sidebar.selectbox("Main Category", options=main_category_options)

# Secondary Category
secondary_category_options = ["BUILDQTY", "UPH_GOAL"]
selected_secondary_category = st.sidebar.selectbox("Secondary Category", options=["Choose an option"] + secondary_category_options)

# Grouped Bar Chart Option
show_grouped_bar = st.sidebar.checkbox("Show as grouped bar chart", key="grouped_bar_checkbox")

# File Uploader
uploaded_file = st.sidebar.file_uploader("Upload a CSV or Excel file", type=['csv', 'xls', 'xlsx'], key="file_uploader")
if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        uploaded_data = pl.read_csv(uploaded_file)
    else:
        uploaded_data = pl.DataFrame(pd.read_excel(uploaded_file))  # Use Pandas to read Excel and convert to Polars DataFrame for compatibility
    st.write("### Uploaded Data")
    st.dataframe(uploaded_data.to_pandas(), use_container_width=True)

# Load data with error handling
build_plan_data_df = load_build_plan_data()
uph_goal_data_df = load_uph_goal_data()
raw_build_plan_data = load_build_plan_for_chart()

# Refresh Button to clear cache and reload data
if st.sidebar.button("Refresh Data", key="refresh_data_button"):
    st.cache_data.clear()
    st.experimental_rerun()

# Function to fetch unique SKUs from the build plan based on SITEID
def fetch_skus_from_build_plan(site_id_filter):
    try:
        connection = create_connection()
        query = f"SELECT DISTINCT ITEM FROM build_plan WHERE SITEID = '{site_id_filter}'"
        data = pd.read_sql_query(query, connection)
        connection.close()
        unique_skus = data['ITEM'].drop_duplicates().tolist()
        st.sidebar.info(f"Total number of unique SKUs fetched: {len(unique_skus)}")
        return unique_skus
    except Exception as e:
        st.error(f"Error loading build plan data: {e}")
        return []

# Base URL for the JSON API endpoint
base_url = "http://mpp-md-dvapp01.corp.sandisk.com:9400/json/SDSM_REST/V_OPERH_AGILE_A162_PROC?IN_SKUNUMBER={IN_SKUNUMBER}"

# Function to build the API URL with SKUs as query parameters
def build_api_url(base_url, sku_list):
    sku_query = ",".join(sku_list)
    return base_url.format(IN_SKUNUMBER=sku_query)

# Function to fetch JSON data from TIBCO DV API using Basic Authentication
def fetch_json_data(url, username, password):
    headers = {'Accept': 'application/json'}
    response = requests.get(url, headers=headers, auth=HTTPBasicAuth(username, password))
    
    if response.status_code == 200:
        return response.json()
    elif response.status_code == 401:
        st.error("Failed to fetch data: Unauthorized (401). Please check your username and password.")
        return None
    else:
        st.error(f"Failed to fetch data from TIBCO DV. Status code: {response.status_code}")
        return None

# Retrieve the username and password securely using Streamlit secrets
try:
    username = st.secrets["TIBCO_USERNAME"]
    password = st.secrets["TIBCO_PASSWORD"]
except KeyError:
    st.error("Username or password not found. Make sure they are set up in your secrets.toml file.")
    username = password = None

# Initialize agile_data_df as an empty DataFrame to avoid NameError
agile_data_df = pl.DataFrame([])

# Process and display the AGILE data if SKUs and credentials are available
sku_list = fetch_skus_from_build_plan('C039')

combined_data = []
if sku_list and username and password:
    batch_size = 100  # Adjust batch size as needed to handle API URL length limits
    for i in range(0, len(sku_list), batch_size):
        batch = sku_list[i:i + batch_size]
        api_url = build_api_url(base_url, batch)
        agile_data = fetch_json_data(api_url, username, password)

        if agile_data and "V_OPERH_AGILE_A162_PROCResponse" in agile_data:
            agile_results = agile_data["V_OPERH_AGILE_A162_PROCResponse"]["V_OPERH_AGILE_A162_PROCResults"]["row"]
            if agile_results:
                agile_df = pd.json_normalize(agile_results)
                
                # Sort SKUs based on LIFECYCLEPHASE in the order C-ACT > C-PROD > DEV
                lifecycle_order = ["C-ACT", "C-PROD", "DEV"]
                agile_df['LIFECYCLEPHASE'] = pd.Categorical(agile_df['LIFECYCLEPHASE'], categories=lifecycle_order, ordered=True)
                agile_df = agile_df.sort_values('LIFECYCLEPHASE')
                
                # Remove duplicates by keeping only the highest-priority phase for each SKU
                agile_df = agile_df.drop_duplicates(subset=['SKUNUMBER'], keep='first')
                
                combined_data.append(agile_df)
if combined_data:
    final_df = pd.concat(combined_data, ignore_index=True)
    agile_data_df = pl.DataFrame(final_df[['SKUNUMBER', 'BD_ID']])

    # Summary message instead of displaying the table
    st.write(f"Total number of unique SKUs in the AGILE DataFrame: {final_df.shape[0]}")
    st.info("AGILE data has been successfully processed and combined.")

    # Download button for unique data
    csv_unique = final_df.to_csv(index=False)
    st.download_button(
        label="Download unique SKUs data as CSV",
        data=csv_unique,
        file_name="unique_agile_data.csv",
        mime="text/csv",
    )
else:
    st.warning("No data found across all batches.")

    st.warning("No SKUs found in the build_plan table for the specified SITEID or credentials are missing.")

# Function to extract the relevant number from BD_ID_from_AGILE
def extract_bd_id_number(build_plan):
    """
    Extract the number after the last '-' from the BD_ID_from_AGILE column
    and create a new column with the extracted value.
    """
    build_plan = build_plan.with_columns(
        pl.col("BD_ID_from_AGILE")
        .str.extract(r"-(\d+)$", group_index=1)  # Regex to capture numbers after the last '-'
        .alias("Extracted_BD_ID")
    )
    return build_plan

# Enrich Build Plan with BD_ID from AGILE Data
def add_bd_id_to_build_plan(build_plan, agile_data):
    build_plan = build_plan.with_columns(
        pl.col("ITEM").cast(pl.Utf8).str.replace(r"^\s+|\s+$", "").str.to_uppercase()
    )
    agile_data = agile_data.with_columns(
        pl.col("SKUNUMBER").cast(pl.Utf8).str.replace(r"^\s+|\s+$", "").str.to_uppercase()
    )
    
    enriched_data = build_plan.join(agile_data, left_on="ITEM", right_on="SKUNUMBER", how="left")
    enriched_data = enriched_data.rename({"BD_ID": "BD_ID_from_AGILE"})
    
    # Apply extraction to create the new column
    if "BD_ID_from_AGILE" in enriched_data.columns:
        enriched_data = extract_bd_id_number(enriched_data)
    
    return enriched_data

# Apply the SITEID filter and enrich the Build Plan
if not agile_data_df.is_empty():
    enriched_build_plan_data = add_bd_id_to_build_plan(build_plan_data_df, agile_data_df)
    if selected_site_ids:
        enriched_build_plan_data = enriched_build_plan_data.filter(pl.col("SITEID").is_in(selected_site_ids))
else:
    enriched_build_plan_data = build_plan_data_df.with_columns(pl.lit(None).alias("BD_ID_from_AGILE"))

# Add UPH column to enriched build plan data by matching Extracted_BD_ID with RECIPE_GROUP
def add_uph_to_build_plan(build_plan, uph_goal):
    # Preprocess RECIPE_GROUP to remove prefixes 'P' and 'Q'
    uph_goal = uph_goal.with_columns(
        pl.col("RECIPE_GROUP").str.replace(r"^[PQ]", "").alias("PROCESSED_RECIPE_GROUP")
    )

    # Preprocess Extracted_BD_ID to match the format
    build_plan = build_plan.with_columns(
        pl.col("Extracted_BD_ID").str.replace(r"^[PQ]", "").alias("PROCESSED_BD_ID")
    )

    # Perform a left join to bring NEW_UPH into the build plan
    enriched_data = build_plan.join(
        uph_goal,
        left_on="PROCESSED_BD_ID",
        right_on="PROCESSED_RECIPE_GROUP",
        how="left"
    )

    # Add the UPH column from NEW_UPH
    enriched_data = enriched_data.with_columns(
        pl.col("NEW_UPH").alias("UPH")
    )

    return enriched_data

# Filter the UPH Goal data for PROCESS == "WB"

def add_uph_to_build_plan(build_plan, uph_goal):
    # Preprocess RECIPE_GROUP in UPH Goal data to ensure uniformity
    uph_goal = uph_goal.with_columns(
        pl.col("BASE_RECIPE").str.replace(r"^[PQ]", "").alias("PROCESSED_RECIPE_GROUP")
    )

    # Preprocess Extracted_BD_ID in Build Plan data to ensure uniformity
    build_plan = build_plan.with_columns(
        pl.col("Extracted_BD_ID").str.replace(r"^[PQ]", "").alias("PROCESSED_BD_ID")
    )

    # Deduplicate UPH Goal data by keeping the row with the lowest NEW_UPH for each RECIPE_GROUP
    uph_goal = uph_goal.sort(["PROCESSED_RECIPE_GROUP", "NEW_UPH"]).unique(subset=["PROCESSED_RECIPE_GROUP"], keep="first")

    # Perform a left join to merge UPH data into the build plan
    enriched_data = build_plan.join(
        uph_goal,
        left_on="PROCESSED_BD_ID",
        right_on="PROCESSED_RECIPE_GROUP",
        how="left"
    )

    # Add the UPH column from NEW_UPH
    enriched_data = enriched_data.with_columns(
        pl.col("NEW_UPH").alias("UPH")
    )

    return enriched_data

# Apply the function to add UPH to the enriched build plan table
if not enriched_build_plan_data.is_empty() and not uph_goal_data_df.is_empty():
    # Filter the UPH Goal data for PROCESS == "WB"
    filtered_uph_goal_data = uph_goal_data_df.filter(pl.col("PROCESS") == "WB")

    # Add the UPH column to the enriched build plan table
    enriched_build_plan_data_with_uph = add_uph_to_build_plan(enriched_build_plan_data, filtered_uph_goal_data)

    # Display the enriched data with UPH
    st.write("### Enriched Build Plan Data with UPH")
    st.dataframe(enriched_build_plan_data_with_uph.to_pandas(), use_container_width=True)

    # Provide a download button for the enriched data with UPH
    csv_enriched_data_with_uph = enriched_build_plan_data_with_uph.to_pandas().to_csv(index=False)
    st.download_button(
        label="Download Enriched Build Plan Data with UPH as CSV",
        data=csv_enriched_data_with_uph,
        file_name="enriched_build_plan_with_uph.csv",
        mime="text/csv",
    )
else:
    st.warning("Unable to add UPH column. Ensure both Build Plan and UPH Goal data are available.")

# Visualization Functions
def show_build_plan_for_chart(build_plan_data_df):
    st.write("## Build Plan Data Visualization")
    required_columns = ["CALENDAR_MONTH", "BUILDQTY", "DEVICE"]
    
    if all(col in build_plan_data_df.columns for col in required_columns):
        build_plan_data_df_pandas = build_plan_data_df.to_pandas()

        fig_build = px.bar(
            build_plan_data_df_pandas.head(10000),
            x="CALENDAR_MONTH",
            y="BUILDQTY",
            color="DEVICE",
            title="Build Quantity by Calendar Month by Device",
            labels={"BUILDQTY": "Build Quantity", "CALENDAR_MONTH": "Calendar Month", "DEVICE": "Device"},
            template="plotly_dark"
        )
        fig_build.update_traces(texttemplate='%{y:.2s}', textposition='outside')
        fig_build.update_layout(yaxis=dict(showgrid=False), xaxis=dict(showgrid=False))
        st.plotly_chart(fig_build, use_container_width=True)
    else:
        st.error(f"Columns {required_columns} are required in the build plan data.")

def show_tool_daily_capacity_chart(data):
    """
    Create and display a bar chart for TOOL_DAILY_CAPACITY.
    """
    # Convert Polars DataFrame to Pandas for compatibility with Plotly
    data_pandas = data.to_pandas()

    # Create a bar chart
    fig = px.bar(
        data_pandas,
        x="RECIPE_GROUP",
        y="TOOL_DAILY_CAPACITY",
        color="RECIPE_GROUP",
        title="Tool Daily Capacity by Recipe Group",
        labels={"TOOL_DAILY_CAPACITY": "Tool Daily Capacity", "RECIPE_GROUP": "Recipe Group"},
        template="plotly_dark"
    )
    fig.update_layout(
        xaxis_title="Recipe Group",
        yaxis_title="Tool Daily Capacity",
        showlegend=False,
        yaxis=dict(showgrid=False),
        xaxis=dict(showgrid=False)
    )
    st.plotly_chart(fig, use_container_width=True)

def show_stacked_area_chart(data):
    required_columns = ["CALENDAR_MONTH", "NODE", "FINANCE_PB_BY_SITE"]

    # Check if the required columns are present
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        st.error(f"The data must contain the following columns: {required_columns}. Missing columns: {missing_columns}")
        st.write("Available columns in data:", list(data.columns))
        return

    # Proceed with the chart if all required columns are present
    data_pandas = data.to_pandas()

    grouped_data = (
        data_pandas.groupby(["CALENDAR_MONTH", "NODE"])["FINANCE_PB_BY_SITE"]
        .sum()
        .reset_index()
    )

    total_by_month = grouped_data.groupby("CALENDAR_MONTH")["FINANCE_PB_BY_SITE"].transform("sum")
    grouped_data["PERCENTAGE"] = (grouped_data["FINANCE_PB_BY_SITE"] / total_by_month) * 100

    # Create the stacked area chart
    fig = px.area(
        grouped_data,
        x="CALENDAR_MONTH",
        y="PERCENTAGE",
        color="NODE",
        title="SIP Technology (PB)",
        labels={
            "PERCENTAGE": "Percentage of Finance PB",
            "CALENDAR_MONTH": "Calendar Month",
            "NODE": "Node",
        },
        template="plotly_white",
    )

    fig.update_layout(
        xaxis_title="Calendar Month",
        yaxis_title="Percentage",
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(0, 101, 10)),
            ticktext=[f"{x}%" for x in range(0, 101, 10)],
        ),
        showlegend=True,
    )

    st.plotly_chart(fig, use_container_width=True)

# Pagination function for displaying data in chunks
def display_data_in_chunks(data, rows_per_page):
    if len(data) == 0:  # Check for empty DataFrame in Polars
        st.warning("No data available to display.")
        return

    total_rows = data.shape[0]
    total_pages = math.ceil(total_rows / rows_per_page)
    selected_page = st.sidebar.selectbox("Select Page", range(1, total_pages + 1), format_func=lambda x: f"Page {x}")
    
    start_row = (selected_page - 1) * rows_per_page
    end_row = start_row + rows_per_page
    st.dataframe(data[start_row:end_row].to_pandas(), use_container_width=True)

# Display enriched Build Plan with BD_ID
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Build Plan", "UPH Goal by Recipe", "Optimal Capacity Plan", "Release Notes", "Developer Resource"])

with tab1:
    st.write("### Build Plan Data with BD_ID")
    if len(enriched_build_plan_data) == 0:
        st.warning("No data available in Build Plan.")
    else:
        # Convert the Enriched Build Plan Data Table to a CSV format
        csv_enriched_data = enriched_build_plan_data.to_pandas().to_csv(index=False)

        # Provide a download button for the table
        st.download_button(
            label="Download Enriched Build Plan Data Table as CSV",
            data=csv_enriched_data,
            file_name="enriched_build_plan_data.csv",
            mime="text/csv",
        )

        # Display the Build Plan chart
        st.write("#### Build Plan Chart")
        show_build_plan_for_chart(enriched_build_plan_data)

      # Display the Stacked Area Chart using raw build_plan data
        st.write("#### Global SiP Loading Forecast")
        if len(raw_build_plan_data) > 0:
            show_stacked_area_chart(raw_build_plan_data)  # Use raw build_plan data
        else:
            st.warning("No data available in the build_plan table for the Stacked Area Chart.")

      # Display the SDE Loading 
        st.write("### SDE Loading by Quarter")
        raw_build_plan_data_sde = load_build_plan_sde()

    if not raw_build_plan_data_sde.is_empty():
        show_sde_loading_chart(raw_build_plan_data_sde)
    else:
        st.warning("No data available for SDE Loading chart.")

with tab2:
    st.write("### UPH Goal by Recipe")
    if len(uph_goal_data_df) == 0:  # Check if the UPH Goal data is empty
        st.warning("No data available in UPH Goal.")  # Display a warning if no data is available
    else:
        # Filter the UPH Goal data by process type "WB"
        filtered_uph_goal_data = uph_goal_data_df.filter(pl.col("PROCESS") == "WB")

        # Check if filtered data is empty
        if len(filtered_uph_goal_data) == 0:
            st.warning("No data available for process type 'WB'.")
        else:
            # Display a message about UPH Goal data being successfully loaded
            st.info("UPH goal data with TOOL_DAILY_CAPACITY and TOOL_MONTHLY_CAPACITY is successfully loaded (Filtered by WB).")

            # Display the filtered UPH Goal data table
            st.write("#### UPH Goal by Recipe (Filtered by WB)")
            st.dataframe(filtered_uph_goal_data.to_pandas(), use_container_width=True)

            # Add a download button for the filtered UPH Goal Data
            csv_data_filtered = filtered_uph_goal_data.to_pandas().to_csv(index=False)
            st.download_button(
                label="Download UPH Goal by Recipe (Filtered by WB) as CSV",
                data=csv_data_filtered,
                file_name="uph_goal_filtered_by_wb.csv",
                mime="text/csv",
            )

            # Add the chart for TOOL_DAILY_CAPACITY
            st.write("#### Tool Daily Capacity Chart (Filtered by WB)")
            show_tool_daily_capacity_chart(filtered_uph_goal_data)


with tab3:
    st.write("### Optimal Capacity Plan")
    st.info("Optimal Capacity Plan data will be displayed here.")

with tab4:
    st.write("### Release Notes")
    st.info("Release notes and updates for this application.")

with tab5:
    st.write("### Developer Resource")
    st.info("Developer resources and settings will be displayed here.")

