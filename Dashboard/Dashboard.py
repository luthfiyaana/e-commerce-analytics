import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

#konfigurasi halaman
st.set_page_config(
    page_title="E-Commerce Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

#CSS custom
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background-color: #FFFFFF;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E9ECE7;
    }

    .brand {
        font-size: 21px;
        font-weight: 700;
        color: #244B3A;
        margin-bottom: 5px;
    }

    .page-heading {
        font-size: 30px;
        font-weight: 700;
        color: #24372D;
        margin-bottom: 4px;
    }

    .page-subtitle {
        color: #818A82;
        font-size: 14px;
        margin-bottom: 24px;
    }

    .section-title {
        font-size: 18px;
        font-weight: 700;
        color: #24372D;
    }

    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E9ECE7;
        border-radius: 15px;
        padding: 20px;
        min-height: 135px;
    }

    .kpi-label {
        color: #7B857D;
        font-size: 13px;
        font-weight: 500;
    }

    .kpi-value {
        color: #24372D;
        font-size: 27px;
        font-weight: 700;
        margin-top: 12px;
    }

    .kpi-caption {
        color: #78906F;
        font-size: 11px;
        margin-top: 5px;
    }

    .insight-card {
        background: #FFFFFF;
        border: 1px solid #E9ECE7;
        border-radius: 15px;
        padding: 20px;
        min-height: 180px;
    }

    .insight-label {
        color: #82907F;
        font-size: 12px;
    }

    .insight-value {
        color: #294E3B;
        font-size: 21px;
        font-weight: 700;
        margin-top: 8px;
    }

    .insight-description {
        color: #7B857D;
        font-size: 12px;
        margin-top: 8px;
        line-height: 1.6;
    }

    div[data-testid="stTabs"] button {
        font-weight: 600;
    }

    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #315B43;
    }

    div[data-testid="stMetric"] {
        background: #FFFFFF;
        padding: 15px;
        border-radius: 12px;
    }

    hr {
        border-color: #E5E9E3;
    }
</style>
""", unsafe_allow_html=True)

#memuat dataset
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


@st.cache_data
def load_data():
    orders = pd.read_csv(DATA_DIR / "orders_dataset.csv")
    order_items = pd.read_csv(DATA_DIR / "order_items_dataset.csv")
    customers = pd.read_csv(DATA_DIR / "customers_dataset.csv")

    return orders, order_items, customers


try:
    orders_df, order_items_df, customers_df = load_data()

except FileNotFoundError as e:
    st.error(
        f"File dataset tidak ditemukan: {e}. "
        "Pastikan folder data berada sejajar dengan dashboard.py "
        "dan berisi file CSV dataset."
    )
    st.stop()

#preprocessing data
orders_df["order_purchase_timestamp"] = pd.to_datetime(
    orders_df["order_purchase_timestamp"],
    errors="coerce"
)

delivered_orders = orders_df[
    orders_df["order_status"] == "delivered"
].copy()

sales_data = delivered_orders.merge(
    order_items_df[["order_id", "price"]],
    on="order_id",
    how="inner"
)

monthly_sales = (
    sales_data.assign(
        month=sales_data["order_purchase_timestamp"].dt.to_period("M")
    )
    .groupby("month")
    .agg(
        total_orders=("order_id", "nunique"),
        total_sales=("price", "sum")
    )
    .reset_index()
)

monthly_sales["month"] = (
    monthly_sales["month"].dt.to_timestamp()
)

total_orders = delivered_orders["order_id"].nunique()

total_sales = sales_data["price"].sum()

total_customers = delivered_orders["customer_id"].nunique()

average_order = total_sales / total_orders if total_orders else 0

#customer RFM analysis
#menggabungkan data pesanan, pelanggan, dan item
rfm_data = delivered_orders[
    ["order_id", "customer_id", "order_purchase_timestamp"]
].merge(
    customers_df[["customer_id", "customer_unique_id"]],
    on="customer_id",
    how="left"
).merge(
    order_items_df[["order_id", "price"]],
    on="order_id",
    how="inner"
)
#menentukan tanggal pengamatan
snapshot_date = (
    delivered_orders["order_purchase_timestamp"].max()
    + pd.Timedelta(days=1)
)
#menghitung Recency, Frequency, dan Monetary
rfm = rfm_data.groupby("customer_unique_id").agg(
    Recency=(
        "order_purchase_timestamp",
        lambda x: (snapshot_date - x.max()).days
    ),
    Frequency=("order_id", "nunique"),
    Monetary=("price", "sum")
).reset_index()
#memberikan skor RFM
rfm["R_score"] = pd.cut(
    rfm["Recency"].rank(method="average", pct=True),
    bins=[0, 0.25, 0.50, 0.75, 1],
    labels=[4, 3, 2, 1],
    include_lowest=True
).astype(int)

rfm["F_score"] = pd.cut(
    rfm["Frequency"],
    bins=[0, 1, 2, 4, float("inf")],
    labels=[1, 2, 3, 4]
).astype(int)

rfm["M_score"] = pd.cut(
    rfm["Monetary"].rank(method="average", pct=True),
    bins=[0, 0.25, 0.50, 0.75, 1],
    labels=[1, 2, 3, 4],
    include_lowest=True
).astype(int)
#menentukan segmen pelanggan
def assign_segment(row):
    r = row["R_score"]
    f = row["F_score"]
    m = row["M_score"]

    if r >= 3 and f >= 3 and m >= 3:
        return "Champions"
    elif r >= 2 and f >= 3:
        return "Loyal Customers"
    elif r >= 3 and f <= 2:
        return "Potential Loyalists"
    elif r <= 2 and f >= 2:
        return "At Risk"
    else:
        return "Others"


rfm["Segment"] = rfm.apply(assign_segment, axis=1)
#ringkasan setiap segmen
rfm_summary = rfm.groupby(
    "Segment", observed=False
).agg(
    Total_Customers=("customer_unique_id", "count"),
    Avg_Recency=("Recency", "mean"),
    Avg_Frequency=("Frequency", "mean"),
    Avg_Monetary=("Monetary", "mean")
).reset_index()
#urutan segmen pada visualisasi
segment_order = [
    "Champions",
    "Loyal Customers",
    "Potential Loyalists",
    "At Risk",
    "Others"
]

rfm_summary["Segment"] = pd.Categorical(
    rfm_summary["Segment"],
    categories=segment_order,
    ordered=True
)

rfm_summary = rfm_summary.sort_values("Segment")

#data wilayah
regional_data = delivered_orders[
    ["order_id", "customer_id"]
].merge(
    customers_df[["customer_id", "customer_state"]],
    on="customer_id",
    how="left"
).merge(
    order_items_df[["order_id", "price"]],
    on="order_id",
    how="inner"
)

state_summary = regional_data.groupby("customer_state").agg(
    Total_Customers=("customer_id", "nunique"),
    Total_Orders=("order_id", "nunique"),
    Total_Sales=("price", "sum")
).reset_index()

state_summary = state_summary.sort_values(
    "Total_Sales", ascending=False
)

state_summary["Sales_Percentage"] = (
    state_summary["Total_Sales"] /
    state_summary["Total_Sales"].sum() * 100
)

#analisis nilai pesanan
order_value = delivered_orders[
    ["order_id"]
].merge(
    order_items_df[["order_id", "price"]],
    on="order_id",
    how="inner"
).groupby("order_id").agg(
    Order_Value=("price", "sum")
).reset_index()

order_value["Order_Category"] = pd.cut(
    order_value["Order_Value"],
    bins=[0, 50, 100, 200, 500, float("inf")],
    labels=[
        "R$0–50",
        "R$50–100",
        "R$100–200",
        "R$200–500",
        "Above R$500"
    ],
    include_lowest=True
)

order_value_summary = order_value.groupby(
    "Order_Category", observed=False
).agg(
    Total_Orders=("order_id", "count"),
    Average_Value=("Order_Value", "mean")
).reset_index()

#sidebar
with st.sidebar:
    st.markdown(
        '<div class="brand">🛒 E-Commerce Analytics</div>',
        unsafe_allow_html=True
    )

    st.caption("BRAZILIAN E-COMMERCE DATA")

    st.divider()

    st.markdown("### Dashboard")
    st.markdown("📊 Sales Overview")
    st.markdown("👥 Customer Analysis")
    st.markdown("🗺️ Regional Analysis")
    st.markdown("💰 Order Value")

    st.divider()

    st.caption("DATA INFORMATION")
    st.caption("Source: Brazilian E-Commerce Dataset")
    st.caption("Analysis: Delivered Orders")

#header
st.markdown(
    '<div class="page-heading">Sales Overview</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="page-subtitle">'
    'Monitor sales performance and understand customer behavior '
    'across Brazilian E-Commerce.'
    '</div>',
    unsafe_allow_html=True
)

#KPI cards
col1, col2, col3, col4 = st.columns(4)

kpi_items = [
    (col1, "TOTAL ORDERS", f"{total_orders:,.0f}", "Delivered orders"),
    (col2, "TOTAL SALES", f"R$ {total_sales:,.2f}", "Product sales value"),
    (col3, "TOTAL CUSTOMERS", f"{total_customers:,.0f}", "Unique customers"),
    (col4, "AVERAGE ORDER", f"R$ {average_order:,.2f}", "Sales per order")
]

for col, label, value, caption in kpi_items:
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-caption">{caption}</div>
        </div>
        """, unsafe_allow_html=True)

st.write("")

#tab analisis
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Sales",
    "👥 Customers",
    "🗺️ Regions",
    "💰 Order Value"
])

#tab 1: sales
with tab1:

    st.markdown(
        '<div class="section-title">Monthly Sales Performance</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Monthly development of orders and product sales."
    )

    chart_col, insight_col = st.columns([2.2, 1])

    with chart_col:

        fig = px.area(
            monthly_sales,
            x="month",
            y="total_sales",
            markers=True,
            labels={
                "month": "Month",
                "total_sales": "Sales (R$)"
            }
        )

        fig.update_traces(
            line_color="#527B60",
            fillcolor="rgba(130, 165, 132, 0.20)",
            marker_color="#527B60",
            hovertemplate=(
                "%{x|%B %Y}<br>"
                "Sales: R$ %{y:,.2f}<extra></extra>"
            )
        )

        fig.update_layout(
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            font_color="#536158",
            margin=dict(l=20, r=20, t=25, b=20),
            height=350,
            xaxis=dict(showgrid=False),
            yaxis=dict(
                showgrid=True,
                gridcolor="#EDF0EB",
                tickprefix="R$ "
            )
        )

        st.plotly_chart(fig, use_container_width=True)

    with insight_col:

        st.markdown(
            '<div class="section-title">Key Insights</div>',
            unsafe_allow_html=True
        )

        if not monthly_sales.empty:

            peak_month = monthly_sales.loc[
                monthly_sales["total_sales"].idxmax()
            ]

            st.markdown(f"""
            <div class="insight-card">
                <div class="insight-label">HIGHEST SALES MONTH</div>
                <div class="insight-value">
                    {peak_month['month'].strftime('%B %Y')}
                </div>
                <div class="insight-description">
                    Recorded sales of R$ {peak_month['total_sales']:,.2f}
                    from {peak_month['total_orders']:,.0f} orders.
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.write("")

            st.markdown(f"""
            <div class="insight-card">
                <div class="insight-label">TOTAL OBSERVATION PERIOD</div>
                <div class="insight-value">
                    {len(monthly_sales)} Months
                </div>
                <div class="insight-description">
                    Monthly sales performance based on delivered orders.
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.write("")

    # Grafik jumlah pesanan bulanan
    st.markdown("#### Monthly Order Volume")

    fig_orders = px.bar(
        monthly_sales,
        x="month",
        y="total_orders",
        labels={
            "month": "Month",
            "total_orders": "Total Orders"
        }
    )

    fig_orders.update_traces(
        marker_color="#9DB69B",
        hovertemplate="%{x|%B %Y}<br>Orders: %{y:,.0f}<extra></extra>"
    )

    fig_orders.update_layout(
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        height=300,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#EDF0EB")
    )

    st.plotly_chart(fig_orders, use_container_width=True)


#tab 2: customer RFM
with tab2:
    st.markdown("### Customer Segmentation")
    st.caption(
        "Customer segmentation based on Recency, Frequency, "
        "and Monetary analysis."
    )

    #ringkasan jumlah pelanggan
    total_rfm_customers = rfm["customer_unique_id"].nunique()

    st.metric(
        label="Total Customers Analyzed",
        value=f"{total_rfm_customers:,.0f}"
    )

    st.write("")

    #grafik 1: Jumlah pelanggan per segmen
    st.markdown("#### Customer Distribution")

    fig_count = px.bar(
        rfm_summary,
        x="Segment",
        y="Total_Customers",
        color="Segment",
        category_orders={"Segment": segment_order},
        color_discrete_sequence=[
            "#527B60",
            "#789A78",
            "#A4B99C",
            "#D0B77C",
            "#AEB6AE"
        ],
        labels={
            "Segment": "Customer Segment",
            "Total_Customers": "Number of Customers"
        }
    )

    fig_count.update_layout(
        showlegend=False,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        height=350,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#EDF0EB")
    )

    st.plotly_chart(fig_count, use_container_width=True)

    #grafik 2-4
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Average Monetary")

        fig_monetary = px.bar(
            rfm_summary,
            x="Segment",
            y="Avg_Monetary",
            color="Segment",
            category_orders={"Segment": segment_order},
            color_discrete_sequence=[
                "#527B60",
                "#789A78",
                "#A4B99C",
                "#D0B77C",
                "#AEB6AE"
            ],
            labels={
                "Segment": "Segment",
                "Avg_Monetary": "Average Monetary (R$)"
            }
        )

        fig_monetary.update_layout(
            showlegend=False,
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            height=320
        )

        st.plotly_chart(
            fig_monetary,
            use_container_width=True
        )

    with col2:
        st.markdown("#### Average Recency")

        fig_recency = px.bar(
            rfm_summary,
            x="Segment",
            y="Avg_Recency",
            color="Segment",
            category_orders={"Segment": segment_order},
            color_discrete_sequence=[
                "#527B60",
                "#789A78",
                "#A4B99C",
                "#D0B77C",
                "#AEB6AE"
            ],
            labels={
                "Segment": "Segment",
                "Avg_Recency": "Average Recency (Days)"
            }
        )

        fig_recency.update_layout(
            showlegend=False,
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            height=320
        )

        st.plotly_chart(
            fig_recency,
            use_container_width=True
        )

    #grafik 4: Frequency
    st.markdown("#### Average Frequency")

    fig_frequency = px.bar(
        rfm_summary,
        x="Segment",
        y="Avg_Frequency",
        color="Segment",
        category_orders={"Segment": segment_order},
        color_discrete_sequence=[
            "#527B60",
            "#789A78",
            "#A4B99C",
            "#D0B77C",
            "#AEB6AE"
        ],
        labels={
            "Segment": "Segment",
            "Avg_Frequency": "Average Frequency (Orders)"
        }
    )

    fig_frequency.update_layout(
        showlegend=False,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        height=320
    )

    st.plotly_chart(
        fig_frequency,
        use_container_width=True
    )

    #tabel ringkasan
    st.markdown("#### RFM Segment Summary")

    display_rfm = rfm_summary.copy()

    display_rfm.columns = [
        "Customer Segment",
        "Total Customers",
        "Average Recency (Days)",
        "Average Frequency",
        "Average Monetary (R$)"
    ]

    st.dataframe(
        display_rfm,
        use_container_width=True,
        hide_index=True
    )

#tab 3: regional analysis
with tab3:
    st.subheader("Analisis Persebaran Pelanggan")
    st.caption(
        "Persebaran pelanggan dan kontribusi penjualan berdasarkan "
        "negara bagian di Brazil."
    )
    #KPI wilayah
    top_state = state_summary.iloc[0]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "JUMLAH NEGARA BAGIAN",
            state_summary["customer_state"].nunique()
        )

    with col2:
        st.metric(
            "PELANGGAN TERBANYAK",
            top_state["customer_state"]
        )

    with col3:
        st.metric(
            "KONTRIBUSI PENJUALAN TERTINGGI",
            f'{top_state["Sales_Percentage"]:.2f}%'
        )

    st.markdown("---")

    #grafik jumlah pelanggan
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Jumlah Pelanggan per Negara Bagian**")

        customer_chart = px.bar(
            state_summary.sort_values(
                "Total_Customers", ascending=True
            ).tail(10),
            x="Total_Customers",
            y="customer_state",
            orientation="h",
            text="Total_Customers",
            labels={
                "Total_Customers": "Jumlah Pelanggan",
                "customer_state": "Negara Bagian"
            }
        )

        customer_chart.update_layout(
            yaxis_title="",
            xaxis_title="Jumlah Pelanggan",
            showlegend=False
        )

        st.plotly_chart(customer_chart, use_container_width=True)

    #grafik nilai penjualan
    with col2:
        st.markdown("**Total Penjualan per Negara Bagian**")

        sales_chart = px.bar(
            state_summary.sort_values(
                "Total_Sales", ascending=True
            ).tail(10),
            x="Total_Sales",
            y="customer_state",
            orientation="h",
            text="Total_Sales",
            labels={
                "Total_Sales": "Total Penjualan (R$)",
                "customer_state": "Negara Bagian"
            }
        )

        sales_chart.update_traces(
            texttemplate="R$ %{x:,.0f}",
            textposition="outside"
        )

        sales_chart.update_layout(
            yaxis_title="",
            xaxis_title="Total Penjualan (R$)",
            showlegend=False
        )

        st.plotly_chart(sales_chart, use_container_width=True)

    #grafik jumlah pesanan
    st.markdown("**Jumlah Pesanan per Negara Bagian**")

    orders_chart = px.bar(
        state_summary.sort_values(
            "Total_Orders", ascending=True
        ).tail(10),
        x="Total_Orders",
        y="customer_state",
        orientation="h",
        text="Total_Orders",
        labels={
            "Total_Orders": "Jumlah Pesanan",
            "customer_state": "Negara Bagian"
        }
    )

    orders_chart.update_layout(
        yaxis_title="",
        xaxis_title="Jumlah Pesanan",
        showlegend=False
    )

    st.plotly_chart(orders_chart, use_container_width=True)

    #tabel ringkasan
    st.markdown("**Ringkasan Data per Negara Bagian**")

    display_state = state_summary.rename(columns={
        "customer_state": "Negara Bagian",
        "Total_Customers": "Jumlah Pelanggan",
        "Total_Orders": "Jumlah Pesanan",
        "Total_Sales": "Total Penjualan (R$)",
        "Sales_Percentage": "Kontribusi Penjualan (%)"
    })

    st.dataframe(
        display_state,
        use_container_width=True,
        hide_index=True
    )
    

#tab 4: order value
with tab4:
    st.subheader("Analisis Nilai Pesanan")
    st.caption(
        "Analisis distribusi nilai pesanan berdasarkan total harga "
        "produk dalam setiap transaksi."
    )
     #KPI
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "RATA-RATA NILAI PESANAN",
            f'R$ {order_value["Order_Value"].mean():,.2f}'
        )

    with col2:
        st.metric(
            "NILAI PESANAN TERTINGGI",
            f'R$ {order_value["Order_Value"].max():,.2f}'
        )

    with col3:
        st.metric(
            "TOTAL PESANAN",
            f'{order_value["order_id"].nunique():,}'
        )

    st.markdown("---")

    #distribusi nilai pesanan
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Distribusi Nilai Pesanan**")

        histogram = px.histogram(
            order_value,
            x="Order_Value",
            nbins=40,
            color_discrete_sequence=["#2E7D5B"],
            labels={
                "Order_Value": "Nilai Pesanan (R$)",
                "count": "Jumlah Pesanan"
            }
        )

        histogram.update_layout(
            xaxis_title="Nilai Pesanan (R$)",
            yaxis_title="Jumlah Pesanan"
        )

        st.plotly_chart(histogram, use_container_width=True)

    with col2:
        st.markdown("**Jumlah Pesanan Berdasarkan Kategori Nilai**")

        category_chart = px.bar(
            order_value_summary,
            x="Order_Category",
            y="Total_Orders",
            color_discrete_sequence=["#2E7D5B"],
            text="Total_Orders",
            labels={
                "Order_Category": "Kategori Nilai Pesanan",
                "Total_Orders": "Jumlah Pesanan"
            }
        )

        category_chart.update_layout(
            xaxis_title="Kategori Nilai Pesanan",
            yaxis_title="Jumlah Pesanan",
            showlegend=False
        )

        st.plotly_chart(category_chart, use_container_width=True)

    #ringkasan
    st.markdown("**Ringkasan Kategori Nilai Pesanan**")

    st.dataframe(
        order_value_summary,
        use_container_width=True,
        hide_index=True
    )

#footer
st.divider()

st.caption(
    "E-Commerce Analytics | Data Analysis Project | "
    "Brazilian E-Commerce Dataset"
)
