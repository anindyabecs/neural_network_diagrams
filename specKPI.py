import streamlit as st
import pandas as pd
import plotly.express as px
import random
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode

# 1. Page Configuration
st.set_page_config(
    page_title="Apollo Dashboard",
    page_icon="🚀",
    layout="wide"
)

# 2. Custom CSS for Styling
st.markdown("""
    <style>
    /* Center text in metrics */
    [data-testid="stMetric"] {
        width: fit-content;
        margin: auto;
    }
    [data-testid="stMetricValue"] {
        text-align: center;
        font-weight: bold;
    }
    [data-testid="stMetricLabel"] {
        text-align: center;
        font-weight: bold;
    }
    /* Custom styling for the Table */
    .stDataFrame {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)


# 3. Mock Data Generator (Updated with new Columns)
@st.cache_data
def get_apollo_data():
    data = []
    statuses = ["Active", "Draft", "Deprecated", "Review"]
    sources = ["CRM DB", "Web Events", "IoT Gateway", "Legacy Mainframe", "Partner API"]
    # Types that map to the requested Transport Mediums
    types = ["JSON", "Avro", "Parquet", "XML", "CSV"]
    downstreams = ["Data Lake", "Fraud Engine", "Marketing Analytics", "Billing System"]
    processing_systems = ["Apollo Core", "Spark Cluster", "AWS Lambda", "Flink Stream"]
    transport_types_technical = ["SFTP", "HTTPS", "JMS", "S3 Bucket", "Kafka Topic"]
    delimiters = [",", "|", ";", "\\t", "Fixed Width"]

    # Database options
    db_types = ["Oracle 19c", "PostgreSQL 14", "MS SQL Server", "MongoDB", "Snowflake"]
    db_names = ["APOLLO_CORE", "CUST_DATA_LGR", "TRANS_HIST_V2", "REF_MASTER", "STAGING_DB"]

    # JMS Options
    jms_topics = ["sales.orders.topic", "inventory.updates.topic", "billing.requests.topic",
                  "shipping.notifications.topic"]
    jms_queues = ["q.order.inbound", "q.inventory.sync", "q.billing.proc", "q.shipping.dispatch"]

    # Generate 30 sample specs
    for i in range(1, 31):
        spec_name = f"Apollo_Spec_{chr(65 + (i % 26))}_Ingest_{i}"
        source = random.choice(sources)
        dtype = random.choice(types)

        # Logic for file-specific fields
        is_file = dtype in ["CSV", "XML", "JSON", "Parquet"]

        # Logic for "Transport Medium" (XML, CSV, DB, JMS) request
        transport_medium = "N/A"
        current_db_type = "N/A"
        current_db_name = "N/A"
        current_jms_topic = "N/A"
        current_jms_queue = "N/A"

        if dtype == "XML":
            transport_medium = "XML"
        elif dtype == "CSV":
            transport_medium = "CSV"
        elif "DB" in source or "Mainframe" in source:
            transport_medium = "DB"
            current_db_type = random.choice(db_types)
            current_db_name = random.choice(db_names)
        else:
            # Fallback for others to match request roughly or assign random including JMS
            transport_medium = random.choice(["DB", "XML", "CSV", "JMS"])
            if transport_medium == "DB":
                current_db_type = random.choice(db_types)
                current_db_name = random.choice(db_names)
            elif transport_medium == "JMS":
                current_jms_topic = random.choice(jms_topics)
                current_jms_queue = random.choice(jms_queues)

        data.append({
            "Data Spec Name": spec_name,
            "Data Spec Description": f"Ingests {dtype} data from {source} for {random.choice(downstreams)}.",
            "Source": source,
            "Type": dtype,
            "Downstream": random.choice(downstreams),
            "Processing System": random.choice(processing_systems),
            "Service Endpoint": f"https://api.apollo.internal/v1/ingest/{1000 + i}",

            # Requested Details Pane Fields
            "Transport Medium": transport_medium,
            "Database Type": current_db_type,
            "Database Name": current_db_name,
            "JMS Topic": current_jms_topic,
            "JMS Queue": current_jms_queue,

            # Existing Columns
            "Transport Type": random.choice(transport_types_technical),
            "XML Path": f"/root/payload/data/v{i}" if transport_medium == "XML" else "N/A",
            "File Path": f"/mnt/apollo/landing/{spec_name.lower()}/" if is_file else "N/A",
            "File Name": f"batch_input_{1000 + i}.{dtype.lower()}" if is_file else "N/A",
            "File Delimiter": random.choice(delimiters) if transport_medium == "CSV" else "N/A",

            # Keeping these for KPIs and Charts logic
            "Status": random.choice(statuses),
            "Rules Associated": random.randint(5, 65),
            "Spec ID": f"SPC-{1000 + i}"
        })
    return pd.DataFrame(data)


# Load data
df = get_apollo_data()

# --- HEADER SECTION ---
col_logo, col_title, col_btn = st.columns([1, 10, 2])

with col_logo:
    # Big Rocket Logo
    st.markdown("## 🚀")

with col_title:
    st.title("Apollo Application Monitor")
    st.write("Real-time oversight of Service Endpoints, Data Specs, and Rules.")

with col_btn:
    if st.button("Refresh Data", type="primary", icon=":material/refresh:"):
        st.rerun()

# --- KPI SECTION ---
with st.container():
    kpi1, kpi2, kpi3 = st.columns(3)

    # KPI 1: Service Endpoint (Count)
    with kpi1:
        with st.container(border=True):
            st.metric(
                label="Service Endpoints",
                value=len(df),  # Dynamic count based on rows
                help="Total active service endpoints."
            )

    # KPI 2: Data Specs
    with kpi2:
        with st.container(border=True):
            active_specs = len(df[df['Status'] == 'Active'])
            st.metric(
                label="Data Specs",
                value=len(df),
                help="The data flow between systems."
            )

    # KPI 3: Total Rules
    with kpi3:
        with st.container(border=True):
            total_rules = df["Rules Associated"].sum()
            st.metric(
                label="Total Rules",
                value=total_rules,
                help="The total number of explicit rules set for the application."
            )

# --- FILTERS & DATA TABLE ---
st.subheader("📋 Data Specification Details")

# Retrieve 'spec_name' from URL query params (e.g. ?spec_name=Apollo_Spec_A)
# If found, it will pre-fill the search box.
url_spec_name = st.query_params.get("spec_name", "")

# Filter Layout
f1, f2, f3 = st.columns([2, 2, 2])
with f1:
    search_query = st.text_input(
        "Search Data Spec Name",
        value=url_spec_name,
        placeholder="Type to search (e.g., Spec_C)..."
    )
with f2:
    status_filter = st.multiselect("Filter by Status", df["Status"].unique(), default=["Active", "Draft"])

# Column Selection Logic (Visible in Grid)
all_columns_display = [
    "Data Spec Name", "Data Spec Description", "Source", "Type",
    "Downstream", "Processing System", "Service Endpoint", "Status",
    "Transport Medium", "Database Type"
]
default_columns = [
    "Data Spec Name", "Data Spec Description", "Source", "Type",
    "Downstream", "Processing System", "Service Endpoint", "Status"
]

with f3:
    selected_columns = st.multiselect(
        "Customize Table Columns",
        options=all_columns_display,
        default=default_columns,
        placeholder="Add/Remove columns..."
    )

# Apply Filters
filtered_df = df.copy()
if status_filter:
    filtered_df = filtered_df[filtered_df["Status"].isin(status_filter)]
if search_query:
    filtered_df = filtered_df[filtered_df["Data Spec Name"].str.contains(search_query, case=False)]

# --- MAIN LAYOUT (GRID) ---
with st.container(border=True):
    # Configure AgGrid
    gb = GridOptionsBuilder.from_dataframe(filtered_df)

    # Configure Selection to be Single Row (Clicking a row triggers selection)
    # Using the code provided: No Checkbox, No Reset Button
    gb.configure_selection('single', use_checkbox=False)

    # Pagination Settings
    gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=10)

    # Hiding columns that aren't selected in the multiselect,
    # BUT keeping them in the dataframe so the Details Pane can access them.
    for col in filtered_df.columns:
        if col not in selected_columns:
            gb.configure_column(col, hide=True)
        else:
            gb.configure_column(col, hide=False)

    gb.configure_side_bar()
    gridOptions = gb.build()

    # Display Grid
    grid_response = AgGrid(
        filtered_df,
        gridOptions=gridOptions,
        enable_enterprise_modules=False,
        height=500,
        theme='streamlit',
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED
    )


# --- POPUP DIALOG LOGIC ---

@st.dialog("🔍 Service Endpoint Details", width="large")
def show_endpoint_details(row):
    # Header section with copyable code blocks arranged horizontally using columns
    # Using vertical_alignment="center" to compact space and improve look

    # Row 1: Service Endpoint Name
    c1, c2 = st.columns([1.5, 3.5], vertical_alignment="center")
    with c1:
        st.markdown("**Service Endpoint Name:**")
    with c2:
        st.code(row.get('Service Endpoint', 'N/A'), language=None)

    # Row 2: Data Spec Name
    c3, c4 = st.columns([1.5, 3.5], vertical_alignment="center")
    with c3:
        st.markdown("**Data Spec Name:**")
    with c4:
        st.code(row.get('Data Spec Name', 'N/A'), language=None)

    st.divider()

    # Split the popup into two columns: Details (Left) and Diagram (Right)
    left_col, right_col = st.columns([1.5, 1], gap="medium")

    # --- LEFT COLUMN: EXISTING DETAILS ---
    with left_col:
        st.markdown("### 📝 Configuration")

        # Determine logic for expanding sections based on Transport Medium
        transport_medium = row.get('Transport Medium', 'N/A')
        is_db = (transport_medium == 'DB')
        is_file = not is_db  # If it's not DB, assume it's file-based (CSV, XML, JMS, etc.)

        # 1. File / Transport Section (Expanded if NOT DB)
        with st.expander("📦 File / Transport Config", expanded=is_file):
            # Using simple markdown lines for Horizontal Key: Value layout
            st.markdown(f"**Medium:** {transport_medium}")

            if transport_medium == 'JMS':
                st.markdown(f"**JMS Topic:** `{row.get('JMS Topic', 'N/A')}`")
                st.markdown(f"**JMS Queue:** `{row.get('JMS Queue', 'N/A')}`")

            st.markdown(f"**File Name:** {row.get('File Name', 'N/A')}")
            st.markdown(f"**File Type:** {row.get('Type', 'N/A')}")
            st.markdown(f"**Delimiter:** `{row.get('File Delimiter', 'N/A')}`")

        # 2. Database Section (Expanded ONLY if DB)
        with st.expander("🗄️ Database Config", expanded=is_db):
            st.markdown(f"**DB Type:** {row.get('Database Type', 'N/A')}")
            st.markdown(f"**DB Name:** {row.get('Database Name', 'N/A')}")

        # 3. XML Path (Expand if Medium is XML, but keeping it simple for now)
        xml_path = row.get('XML Path', 'N/A')
        if xml_path != "N/A":
            with st.expander("XML Details", expanded=(transport_medium == 'XML')):
                # Inline code block for horizontal alignment
                st.markdown(f"**XML Path:** `{xml_path}`")

    # --- RIGHT COLUMN: FLOW DIAGRAM ---
    with right_col:
        st.markdown("### 🔀 Data Flow")

        # Wrapped in a bordered container
        with st.container(border=True):
            st.write("")  # Spacer

            # Define SVG for a clean Visio-style vertical arrow
            visio_arrow_html = """
            <div style="text-align: center; margin: -5px 0;">
                <svg width="24" height="40" viewBox="0 0 24 40" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <!-- Start Circle (Connection Point) -->
                    <circle cx="12" cy="4" r="3" fill="#adb5bd"/>
                    <!-- Straight Vertical Line -->
                    <line x1="12" y1="4" x2="12" y2="34" stroke="#adb5bd" stroke-width="2"/>
                    <!-- Arrow Head -->
                    <path d="M7 32 L 12 40 L 17 32 L 7 32 Z" fill="#adb5bd"/>
                </svg>
            </div>
            """

            # 1. Source Node (Darker Gradient Grey)
            st.markdown(f"""
            <div style="text-align: center; border: 2px solid #ddd; padding: 12px; border-radius: 8px; background: linear-gradient(to bottom, #e2e6ea, #dae0e5);">
                <strong style="color: #6c757d; font-size: 0.9em;">SOURCE</strong><br>
                <span style="color: #007BFF; font-weight: bold; font-size: 1.1em;">{row.get('Source', 'Unknown')}</span>
            </div>
            """, unsafe_allow_html=True)

            # Visio Arrow
            st.markdown(visio_arrow_html, unsafe_allow_html=True)

            # 2. Endpoint Node (Darker Gradient Pink)
            st.markdown(f"""
            <div style="text-align: center; border: 2px solid #d63384; padding: 12px; border-radius: 8px; background: linear-gradient(to bottom, #ffdae0, #ffc2cd);">
                <strong style="color: #d63384; font-size: 0.9em;">SERVICE ENDPOINT</strong><br>
                <span style="font-size: 0.85em; word-break: break-all; color: #333;">{row.get('Service Endpoint', 'N/A')}</span>
            </div>
            """, unsafe_allow_html=True)

            # Visio Arrow
            st.markdown(visio_arrow_html, unsafe_allow_html=True)

            # 3. Processing System Node (Darker Gradient Orange)
            st.markdown(f"""
            <div style="text-align: center; border: 2px solid #fd7e14; padding: 12px; border-radius: 8px; background: linear-gradient(to bottom, #ffe5d0, #ffcca0);">
                <strong style="color: #fd7e14; font-size: 0.9em;">PROCESSING SYSTEM</strong><br>
                <span style="font-size: 0.9em; font-weight: bold; color: #333;">{row.get('Processing System', 'N/A')}</span>
            </div>
            """, unsafe_allow_html=True)

            # Visio Arrow
            st.markdown(visio_arrow_html, unsafe_allow_html=True)

            # 4. Destination Node (Darker Gradient Green)
            st.markdown(f"""
            <div style="text-align: center; border: 2px solid #28a745; padding: 12px; border-radius: 8px; background: linear-gradient(to bottom, #d4edda, #c3e6cb);">
                <strong style="color: #28a745; font-size: 0.9em;">DESTINATION</strong><br>
                <span style="color: #155724; font-weight: bold; font-size: 1.1em;">{row.get('Downstream', 'Unknown')}</span>
            </div>
            """, unsafe_allow_html=True)

            st.write("")  # Bottom spacer


# Selection Handling
selected = grid_response['selected_rows']

# Initialize session state for tracking last selected item
if "last_selected_spec_id" not in st.session_state:
    st.session_state.last_selected_spec_id = None

if selected is not None and len(selected) > 0:
    # Handle DataFrame vs List return
    if isinstance(selected, pd.DataFrame):
        row = selected.iloc[0].to_dict()
    else:
        row = selected[0]

    current_spec_id = row.get("Spec ID")

    # Logic: Only open the dialog if the selection has CHANGED since the last run.
    # This prevents the dialog from reopening instantly after you close it.
    if current_spec_id != st.session_state.last_selected_spec_id:
        st.session_state.last_selected_spec_id = current_spec_id
        show_endpoint_details(row)

else:
    # Reset selection if table is cleared
    st.session_state.last_selected_spec_id = None

st.divider()

# --- ANALYTICS CHARTS SECTION ---
st.subheader("📊 Analytics Overview")

row1_col1, row1_col2 = st.columns([2, 1])

# Chart 1: Bar Chart (Complexity Analysis)
with row1_col1:
    with st.container(border=True):
        st.markdown("##### Top Complex Specifications")
        chart_df = filtered_df.nlargest(10, "Rules Associated")

        fig_bar = px.bar(
            chart_df,
            x="Data Spec Name",
            y="Rules Associated",
            color="Rules Associated",
            color_continuous_scale="Blues",
            text_auto=True
        )
        fig_bar.update_layout(height=350, margin=dict(t=10, b=10))
        st.plotly_chart(fig_bar, use_container_width=True)

# Chart 2: Donut Chart (Status Distribution)
with row1_col2:
    with st.container(border=True):
        st.markdown("##### Status Distribution")

        status_counts = filtered_df["Status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]

        fig_donut = px.pie(
            status_counts,
            values="Count",
            names="Status",
            hole=0.5,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_donut.update_layout(height=350, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_donut, use_container_width=True)
