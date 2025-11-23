# Exploratory Data Analysis
.Understanding the dataset to explore how the data is present in the database and if there is a need of creating some aggrigated that can help with:
.Vendor Selection for profibility
.Product Pricing Optimization

import pandas as pd

import sqlite3

# Creating Database Connection
conn = sqlite3.connect('inventory.db')

# Checking tabls present in database
tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type = 'table'", conn)
tables

pd.read_sql("select count(*) from purchases", conn)

for table in tables ['name']:
    print('-'*50, f'{table}', '-'*50)
    print("Count of record:", pd.read_sql(f"select count(*) as count from {table}", conn) ["count"].values[0])
    display(pd.read_sql(f"select * from {table} limit 6",conn))

purchases = pd.read_sql("select * from purchases WHERE VendorNumber = 4466", conn)
purchases

purchase_prices = pd.read_sql("""select * from purchase_prices WHERE VendorNumber = 4466""", conn)
purchase_prices

vendor_invoice = pd.read_sql("""select * from vendor_invoice WHERE VendorNumber = 4466""", conn)
vendor_invoice

sales = pd.read_sql("""select * from sales WHERE VendorNo = 4466""", conn)
sales

purchases.groupby(['Brand','PurchasePrice'])[['Quantity','Dollars']].sum()

vendor_invoice['PONumber'].nunique()
vendor_invoice.shape
vendor_invoice.columns
purchases.groupby(['Brand','PurchasePrice'])[['Quantity','Dollars']].sum()

sales.groupby('Brand')[['SalesDollars', 'SalesPrice', 'SalesQuantity']].sum()

.The purchases table contains actual purchase data, including the data of purchase, products (brands) purchased by vendor, the amount paid (in dollars),and the quantity purchased
.The purchase price column is derived from the purchase_price table, which provides product_wise actual and purchase prices. The combination of vendor and brand is unique in this table
.The vendor_invoice table aggregates data from the purchases table, summarixing quantity and dollar amounts, along with an additional column for freight, This table maintains uniqueness based on vendor and PO number.
.This sales table captures actual sales transactions, detailing the brands purchased by vendors, the quantity sold, the selling price, and the revenue earned.
# As the data that we need for analysis is distributed in different tables, we need to create a summary table containing:
.Purchase transactions made by vendors
.Sales transaction data
.Freight costs for each vendor
.Actual product prices from vendors

freight_summary = pd.read_sql_query("""select VendorNumber, SUM(Freight) as FreightCost
from vendor_invoice
GROUP BY VendorNumber""", conn)
freight_summary

pd.read_sql_query("""select
            p.VendorNumber,
            p.VendorName,
            p.PurchasePrice,
            p.Brand,
            pp.Volume,
            pp.Price as ActualPrice,
            SUM(p.Quantity) as TotalPurchaseQuantity,
            SUM(p.Dollars) as TotalPurchaseDollars
            FROM purchases p
            JOIN purchase_prices pp
            ON p.Brand = pp.Brand
            WHERE p.PurchasePrice>0
            GROUP BY P.VendorNumber, p.VendorName, p.Brand
            ORDER BY TotalPurchaseDollars""", conn)
pd.read_sql_query("""SELECT
            VendorNo,
            Brand,
            SUM(SalesDollars) as TotalSalesDollars,
            SUM(SalesPrice) as TotalSalesPrice,
            SUM(SalesQuantity) as TotalSalesQuantity,
            SUM(ExciseTax) as TotalExciseTax
            FROM sales
            GROUP BY VendorNo, Brand
            ORDER BY TotalSalesDollars""", conn)
vendor_sales_summary = pd.read_sql_query("""WITH FreightSummary AS(
            SELECT
                VendorNumber,
                SUM(Freight) AS FreightCost
            FROM vendor_invoice
            GROUP BY VendorNumber
),

PurchaseSummary AS (
        SELECT
            p.VendorNumber,
            p.VendorName,
            p.Brand,
            p.Description,
            p.PurchasePrice,
            pp.Price AS ActualPrice,
            pp.Volume,
            SUM(p.Quantity) AS TotalPurchaseQuantity,
            SUM(p.Dollars) as TotalPurchaseDollars
        FROM purchases p
        JOIN purchase_prices pp
            ON p.Brand = pp.Brand
        WHERE p.PurchasePrice > 0
        GROUP BY p.VendorNumber, p.VendorName, p.Brand, p.Description, p.PurchasePrice, pp.Price, pp.Volume
),

SalesSummary AS (
        SELECT
            VendorNo,
            Brand,
            SUM(SalesQuantity) AS TotalSalesQuantity,
            SUM(SalesDollars) AS TotalSalesDollars,
            SUM(SalesPrice) AS TotalSalesPrice,
            SUM(ExciseTax) AS TotalExciseTax
        FROM sales
        GROUP BY VendorNo, Brand
)

SELECT
    ps.VendorNumber,
    ps.VendorName,
    ps.Brand,
    ps.Description,
    ps.PurchasePrice,
    ps.ActualPrice,
    ps.Volume,
    ps.TotalPurchaseQuantity,
    ps.TotalPurchaseDollars,
    ss.TotalSalesQuantity,
    ss.TotalSalesDollars,
    ss.TotalSalesPrice,
    ss.TotalExciseTax,
    fs.FreightCost
FROM PurchaseSummary ps
LEFT JOIN SalesSummary ss
    ON ps.VendorNumber = ss.VendorNo
    AND ps.Brand = ss.Brand
LEFT JOIN FreightSummary fs
    ON ps.VendorNumber = fs.VendorNumber
ORDER BY ps.TotalPurchaseDollars DESC""", conn)

vendor_sales_summary


# This query generates a vendor-wise sales and purchase summary, which is valuable for:
# Performance Optimization:
.The query involves heavy joins and aggregations on large datasets like sales and purchases.
.Storing the pre-aggregated results avoids repeated expensive computations.
.Helps in analyzing sales, purchases, and pricing for different vendors and brands.
.Future benefits of storing this data for faster dashboarding and reporting.
.Instead of running expensive queries each time, dashboards can fetch data quickly from vendor_sales_summary.

vendor_sales_summary['Description'].unique()
vendor_sales_summary['VendorName'].unique()
vendor_sales_summary['Volume'] =  vendor_sales_summary['Volume'].astype('float64')
vendor_sales_summary.fillna(0, inplace = True)
vendor_sales_summary['VendorName'] = vendor_sales_summary['VendorName'].str.strip()
vendor_sales_summary['GrossProfit'] = vendor_sales_summary['TotalSalesDollars'] - vendor_sales_summary['TotalPurchaseDollars']
vendor_sales_summary

vendor_sales_summary['ProfitMargin'] = (vendor_sales_summary['GrossProfit'] / vendor_sales_summary['TotalSalesDollars'])*100
vendor_sales_summary
vendor_sales_summary['StockTurnover'] = vendor_sales_summary['TotalSalesQuantity'] / vendor_sales_summary['TotalPurchaseQuantity']
vendor_sales_summary

vendor_sales_summary['SalestoPurchaseRatio'] = vendor_sales_summary['TotalSalesDollars'] / vendor_sales_summary['TotalPurchaseDollars']
vendor_sales_summary
vendor_sales_summary.columns

cursor = conn.cursor()
cursor.execute("DROP TABLE IF EXISTS vendor_sales_summary;")
cursor.execute("""
CREATE TABLE vendor_sales_summary (
    VendorNumber INT,
    VendorName VARCHAR(100),
    Brand INT,
    Description VARCHAR(100),
    PurchasePrice DECIMAL(10,2),
    ActualPrice DECIMAL(10,2),
    Volume DECIMAL(10,2),
    TotalPurchaseQuantity INT,
    TotalPurchaseDollars DECIMAL(10,2),
    TotalSalesQuantity INT,
    TotalSalesDollars DECIMAL(15,2),
    TotalSalesPrice DECIMAL(15,2),
    TotalExciseTax DECIMAL(15,2),
    FreightCost DECIMAL(15,2),
    GrossProfit DECIMAL(15,2),
    ProfitMargin DECIMAL(15,2),
    StockTurnover DECIMAL(15,2),
    SalestoPurchaseRatio DECIMAL(15,2),
    PRIMARY KEY (VendorNumber, Brand)
);
""")

pd.read_sql_query("select * from vendor_sales_summary", conn)
vendor_sales_summary.to_sql('vendor_sales_summary', conn, if_exists = 'replace', index = False)
pd.read_sql_query("select * from vendor_sales_summary", conn)
vendor_sales_summary.columns

import sqlite3
import pandas as pd
import logging
from ingestion_db import ingest_db

logging.basicConfig(
    filename = "logs/get_vendor_summary.log",
    level = logging.DEBUG,
    format = "%(asctime)s - %(levelname)s - %(message)s",
    filemode = "a"
)

def create_vendor_summary(conn):
    '''This function will merge the deffirent tables to get the overall vendor summary and adding new columns in the resultant data'''
    vendor_sales_summary = pd.read_sql_query("""WITH FreightSummary AS(
            SELECT
                VendorNumber,
                SUM(Freight) AS FreightCost
            FROM vendor_invoice
            GROUP BY VendorNumber
    ),

    PurchaseSummary AS (
        SELECT
            p.VendorNumber,
            p.VendorName,
            p.Brand,
            p.Description,
            p.PurchasePrice,
            pp.Price AS ActualPrice,
            pp.Volume,
            SUM(p.Quantity) AS TotalPurchaseQuantity,
            SUM(p.Dollars) as TotalPurchaseDollars
        FROM purchases p
        JOIN purchase_prices pp
            ON p.Brand = pp.Brand
        WHERE p.PurchasePrice > 0
        GROUP BY p.VendorNumber, p.VendorName, p.Brand, p.Description, p.PurchasePrice, pp.Price, pp.Volume
        ),

        SalesSummary AS (
            SELECT
                VendorNo,
                Brand,
                SUM(SalesQuantity) AS TotalSalesQuantity,
                SUM(SalesDollars) AS TotalSalesDollars,
                SUM(SalesPrice) AS TotalSalesPrice,
                SUM(ExciseTax) AS TotalExciseTax
            FROM sales
            GROUP BY VendorNo, Brand
        )

        SELECT
        ps.VendorNumber,
        ps.VendorName,
        ps.Brand,
        ps.Description,
        ps.PurchasePrice,
        ps.ActualPrice,
        ps.Volume,
        ps.TotalPurchaseQuantity,
        ps.TotalPurchaseDollars,
        ss.TotalSalesQuantity,
        ss.TotalSalesDollars,
        ss.TotalSalesPrice,
        ss.TotalExciseTax,
        fs.FreightCost
    FROM PurchaseSummary ps
    LEFT JOIN SalesSummary ss
        ON ps.VendorNumber = ss.VendorNo
        AND ps.Brand = ss.Brand
    LEFT JOIN FreightSummary fs
        ON ps.VendorNumber = fs.VendorNumber
    ORDER BY ps.TotalPurchaseDollars DESC""", conn)

    return vendor_sales_summary


def clean_data(df):
    '''This function will clear the data'''
    # Changing data type to float
    df['Volume'] =  df['Volume'].astype('float')

    # Filling missing value with 0
    df.fillna(0, inplace=True)

    # Removing spaces from categorical columns
    df['VendorName'] = df['VendorName'].str.strip()
    df['Description'] = df['Description'].str.strip()

    # Creating new columns for better analysis
    df['GrossProfit'] = df['TotalSalesDollars'] - df['TotalPurchaseDollars']
    df['ProfitMargin'] = (df['GrossProfit'] / df['TotalSalesDollars']) * 100
    df['StockTurnover'] = df['TotalSalesQuantity'] / df['TotalPurchaseQuantity']
    df['SalestoPurchaseRatio'] = df['TotalSalesDollars'] / df['TotalPurchaseDollars']

    return df

if __name__ == '__main__':
    
    # Creating database connection
    conn = sqlite3.connect('inventory.db')

    logging.info('Creating vendor summary table......')
    summary_df = create_vendor_summary(conn)
    logging.info(summary_df.head())

    logging.info('Cleaning Data..........')
    clean_df = clean_data(summary_df)
    logging.info(clean_df.head())

    logging.info('ingestion data.......')
    ingest_db(clean_df, 'vendor_sales_summary', conn)
    logging.info('Completed')


# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sqlite3
from scipy.stats import ttest_ind
import scipy.stats as stats
warnings.filterwarnings('ignore')

# Loading the Dataset
# Creating database connection
conn = sqlite3.connect('inventory.db')
# Feching vendor summary data
df = pd.read_sql_query("select * from vendor_sales_summary",conn)
df.head()

# Exploratory Data Analysis
.Previously, we examined the various tables in the database to identify key varaibles, understand their relationship, determine which one should be included in the final analysis.
.In this phase of EDA, We will analyze the resultant table to gain insight into the distribution of each column. This will help us understand data patterns, identify anomalies, and ensure data quality before proceeding with further analysis.# Outlier Detection with Boxplots
plt.figure(figsize=(15,10))
for i, col in enumerate(numerical_cols):
    plt.subplot(4, 4, i+1)
    sns.boxplot(y=df[col])
    plt.title(col)
plt.tight_layout()
plt.show()
# Distribution plots for Numerical columns
numerical_cols = df.select_dtypes(include=np.number).columns

plt.figure(figsize=(15,10))
for i, col in enumerate(numerical_cols):
    plt.subplot(4, 4, i+1)  # Adjust grid layout as needed
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(col)
plt.tight_layout()
plt.show()

# Summary Statistics INSIGHT
.Negative & Zero Values
Gross Profit: Min value is 52002.78, indicating losses. Some products or transactions may be selling at a loss due to high costs or selling at discount lower than the purchase price. -Profit Margin: Has a min of -, which suggests cases where revenue is zero or even than costs. -Total sales Quantity & sales Dollars: Min value are 0, meaning some products were purchased but never sold. These could be slow_moving or obsulete stock.
Outlier Indicated by High Standard Deviation
Purchase & Actual Prices: The max values(5681.81 & 7499.99) are significantly higher than the mean (24.39 & 35.64), indicating potential premium products.
Freight Cost: Huge variation, from 0.09 to 257032.07, suggests logistics inefficiencies or bulk shipments.
Stock Turnover: Ranges 0 to 274.5, implying some products cell extremely fast while others remain in stock indefinitely. Value more than 1 indicates that sold quantity for that product is higher than purchased quantity due to either sales are being fulfilled from older stock.

# Let's filter the data by removing inconsistencies
df = pd.read_sql_query("""SELECT *
FROM vendor_sales_summary
WHERE GrossProfit > 0
AND ProfitMargin > 0
AND TotalSalesQuantity > 0""", conn)

# Distribution plots for Numerical columns
numerical_cols = df.select_dtypes(include=np.number).columns

plt.figure(figsize=(15,10))
for i, col in enumerate(numerical_cols):
    plt.subplot(4, 4, i+1)  # Adjust grid layout as needed
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(col)
plt.tight_layout()
plt.show()

# Count plots for categorical columns
categorical_cols = ["VendorName", "Description"]
plt.figure(figsize=(15,10))
for i, col in enumerate(categorical_cols):
    plt.subplot(1, 2, i+1)
    sns.countplot(y=df[col], order=df[col].value_counts().index[:10]) # Top 10 categories
    plt.title(f"Count Plot of {col}")
plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12,8))
correlation_matrix = df[numerical_cols].corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# Correlation Insights
PurchasePrice has weak correlations with TotalSalesDollars (-0.012) and GrossProfit (-0.016), suggesting that price variations do not significantly impact sales revenue or profit.
Strong correlation between total purchase quantity and total sales quantity (0.999), confirming efficient inventory turnover.
Negative correlation between profit margin and total sales price (-0.179) suggests that as sales price increases, margins decrease, possibly due to competitive pricing pressures.
StockTurnover has weak negative correlations with both GrossProfit (-0.038) and ProfitMargin (-0.055), indicating that faster turnover does not necessarily result in higher profitability.
# Data Analysis
Identify Brands that need Promotional or Pricing Adjustments which exhibit lower sales performance but higher profit margins.

# df.groupby('Description').agg({
#     'TotalSalesDollars':'sum',
#     'ProfitMargin':'mean'})
brand_performance=df.groupby('Description').agg({
    'TotalSalesDollars':'sum',
    'ProfitMargin':'mean'}).reset_index()
brand_performance

low_sales_threshold=brand_performance['TotalSalesDollars'].quantile(0.15)
high_margin_threshold=brand_performance['ProfitMargin'].quantile(0.85)

# Filter brand with low sales but high profit margins
target_brands = brand_performance[
        (brand_performance['TotalSalesDollars'] <= low_sales_threshold) &
        (brand_performance['ProfitMargin'] >= high_margin_threshold)
]
print("Brand with low sales but high profit margins:")
display(target_brands.sort_values('TotalSalesDollars'))

# For better visualization
brand_performance = brand_performance[brand_performance['TotalSalesDollars']<10000] 

plt.figure(figsize=(10, 6))
sns.scatterplot(data=brand_performance, x='TotalSalesDollars', y='ProfitMargin', color="blue", label="All Brands", alpha=0.2)
sns.scatterplot(data=target_brands, x='TotalSalesDollars', y='ProfitMargin', color="red", label="Target Brands")

plt.axhline(high_margin_threshold, linestyle='--', color='black', label="High Margin Threshold")
plt.axvline(low_sales_threshold, linestyle='--', color='black', label="Low Sales Threshold")

plt.xlabel("Total Sales ($)")
plt.ylabel("Profit Margin (%)")
plt.title("Brands for Promotional or Pricing Adjustments")
plt.legend()
plt.grid(True)
plt.show()

# Which Vendors and Brands demonstrats the highest sales performance
def format_dollars(value):
    if value >= 1_000_000:
        return f"{value/1_000_000:.2f}M"
    elif value >= 1_000:
        return f"{value/1_000:.2f}K"
    else:
        return str(value)
# Top Vendors & brands by sales performance
top_vendors = df.groupby('VendorName')['TotalSalesDollars'].sum().nlargest(10)
top_brands = df.groupby('Description')['TotalSalesDollars'].sum().nlargest(10)    

plt.figure(figsize=(15, 5))

# Plot for Top Vendors
plt.subplot(1, 2, 1)
ax1 = sns.barplot(y=top_vendors.index, x=top_vendor.values, palette="Blues_r")
plt.title("Top 10 Vendors by Sales")

for bar in ax1.patches:
    ax1.text(bar.get_width() + (bar.get_width() * 0.02),
             bar.get_y() + bar.get_height() / 2,
             format_dollars(bar.get_width()),
             ha='left', va='center', fontsize=10, color='black')

# Plot for Top Brands
plt.subplot(1, 2, 2)
ax2 = sns.barplot(y=top_brands.index.astype(str), x=top_brand.values, palette="Reds_r")
plt.title("Top 10 Brands by Sales")

for bar in ax2.patches:
    ax2.text(
        bar.get_width() + (bar.get_width() * 0.02),
        bar.get_y() + bar.get_height() / 2,
        format_dollars(bar.get_width()),
        ha='left', va='center', fontsize=10, color='black'
    )
plt.tight_layout()
plt.show()

# Which Vendor contribute the most to total purchase dollars

vendor_performance=df.groupby('VendorName').agg({
    'TotalPurchaseDollars':'sum',
    'GrossProfit':'sum',
    'TotalSalesDollars':'sum'
}).reset_index()
vendor_performance['Purchase_Contribution%'] = vendor_performance['TotalPurchaseDollars']/vendor_performance['TotalPurchaseDollars'].sum()*100
vendor_performance=round(vendor_performance.sort_values('Purchase_Contribution%', ascending = False),2)
# Display top 10 vendors
top_vendors = vendor_performance.head(10)
# top_vendors['TotalSalesDollar'] = top_vendors['TotalSalesDollars'].apply(format_dollars)
top_vendors['TotalPurchaseDollars'] = top_vendors['TotalPurchaseDollars'].apply(format_dollars)
top_vendors['GrossProfit'] = top_vendors['GrossProfit'].apply(format_dollars)
top_vendors

top_vendors['Cumulative_Contribution%'] = top_vendors['Purchase_Contribution%'].cumsum()
top_vendors

fig, ax1 = plt.subplots(figsize=(10, 6))

# Bar plot for Purchase Contribution%
sns.barplot(x=top_vendors['VendorName'], y=top_vendors['Purchase_Contribution%'], palette="mako", ax=ax1)

for i, value in enumerate(top_vendors['Purchase_Contribution%']):
    ax1.text(i, value - 1, str(value) + '%', ha='center', fontsize=10, color='white')

# Line plot for Cumulative Contribution%
ax2 = ax1.twinx()
ax2.plot(top_vendors['VendorName'], top_vendors['Cumulative_Contribution%'],
         color='red', marker='o', linestyle='dashed', label='Cumulative Contribution %')

ax1.set_xticklabels(top_vendors['VendorName'], rotation=90)
ax1.set_ylabel('Purchase Contribution %', color='blue')
ax2.set_ylabel('Cumulative Contribution %', color='red')
ax1.set_xlabel('Vendors')
ax1.set_title('Pareto Chart: Vendor Contribution to Total Purchases')

ax2.axhline(y=100, color='gray', linestyle='dashed', alpha=0.7)
ax2.legend(loc='upper right')

plt.show()

# How much of total procurement is dependent on the tops vendors
print(f"Total purchase contribution of top 10 vendors is {round(top_vendors['Purchase_Contribution%'].sum(),2)}%")
vendors = list(top_vendors['VendorName'].values)
purchase_contributions = list(top_vendors['Purchase_Contribution%'].values)
total_contribution = sum(purchase_contributions)
remaining_contribution = 100 - total_contribution

# Append "Others vendors" category
vendors.append("Other Vendors")
purchase_contributions.append(remaining_contribution)

# Donut chart
fig, ax = plt.subplots(figsize=(8, 8))
wedges, texts, autotexts = ax.pie(purchase_contributions, labels=vendors, autopct='%1.1f%%', startangle=140, pctdistance=0.85, 
                                  colors=plt.cm.Paired.colors)

# Draw a white circle in the center to center a "donut" effect
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig.gca().add_artist(centre_circle)

# Add total contribution annotation in the center
plt.text(0, 0, f"Top 10 Total:\{total_contribution:.2f}%", fontsize=14, fontweight='bold', ha='center', va='center')

plt.title("Top 10 vendor's purchase Contribution (%)")
plt.show()

# Does purchasing is bulk reduce the unit price, and what is the optimal purchase volume for cost savings?
df['UnitPurchasePrice']=df['TotalPurchaseDollars']/df['TotalPurchaseQuantity']
df['OrderSize'] = pd.qcut(df['TotalPurchaseQuantity'], q=3, labels=["Small", "Medium", "Larg"])

df[['OrderSize', 'TotalPurchaseQuantity']]
df.groupby('OrderSize')[['UnitPurchasePrice']].mean()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="OrderSize", y="UnitPurchasePrice", palette="Set2")
plt.title("Impact of bulk purchasing on Unit Price")
plt.xlabel("OrderSize")
plt.ylabel("Average Unit purchase Price")
plt.show()

.Vendors buying in bulk (Larg order size) get the lowest unit price ($10.78 per unit), meaning higher margins if they can manage inventory efficiently.
.The price different between Small and Larg orders is subsstantial (~72% reduction in unit cost).
.This suggests that bulk pricing strategies successfully encourage vendors to purchase in larg volumes, leading to higher overall sales despite lower per_unit revenue.

# Which vendors have low inventory turnover, indicating access stock and slow_moving products?
df[df['StockTurnover']<1].groupby('VendorName')[['StockTurnover']].mean().sort_values('StockTurnover', ascending=True).head(10)
How much capital is locked in unsold inventory per vendor, and which vendors contribute the most to it?
df['UnsoldInventoryValue'] = (df['TotalPurchaseQuantity'] - df['TotalSalesQuantity']) * df['PurchasePrice']
print('Total Unsold Capital:', format_dollars(df['UnsoldInventoryValue'].sum()))
What is the 95% confidence interals for profit margins of top_performing and low_performing vendors.

top_threshold = df['TotalSalesDollars'].quantile(0.75)
low_threshold = df['TotalSalesDollars'].quantile(0.25)
top_vendors = df[df['TotalSalesDollars'] >= top_threshold]['ProfitMargin'].dropna()
low_vendors = df[df['TotalSalesDollars'] <= low_threshold]['ProfitMargin'].dropna()

def confidence_interval(data, confidence=0.95):
    mean_val = np.mean(data)
    std_err = np.std(data, ddof=1) / np.sqrt(len(data)) #Standard Error
    t_critical = stats.t.ppf((1 + confidence) / 2, df=len(data) -1 )
    margin_of_error = t_critical * std_err
    return mean_val, mean_val - margin_of_error, mean_val + margin_of_error
Calculate confidence intervals for top and low vendors
top_mean, top_lower, top_upper = confidence_interval(top_vendors)
low_mean, low_lower, low_upper = confidence_interval(low_vendors)

# Print confidence intervals and means
print(f"Top Vendors 95% CI: ({top_lower:.2f}, {top_upper:.2f}), Mean: {top_mean:.2f}")
print(f"Low Vendors 95% CI: ({low_lower:.2f}, {low_upper:.2f}), Mean: {low_mean:.2f}")

plt.figure(figsize=(12, 6))

# Top Vendors Plot
sns.histplot(top_vendors, kde=True, color="blue", bins=30, alpha=0.5, label="Top Vendors")
plt.axvline(top_lower, color="blue", linestyle="--", label=f"Top Lower: {top_lower:.2f}")
plt.axvline(top_upper, color="blue", linestyle="--", label=f"Top Upper: {top_upper:.2f}")
plt.axvline(top_mean, color="blue", linestyle="-", label=f"Top Mean: {top_mean:.2f}")

# Low Vendors Plot
sns.histplot(low_vendors, kde=True, color="red", bins=30, alpha=0.5, label="Low Vendors")
plt.axvline(low_lower, color="red", linestyle="--", label=f"Low Lower: {low_lower:.2f}")
plt.axvline(low_upper, color="red", linestyle="--", label=f"Low Upper: {low_upper:.2f}")
plt.axvline(low_mean, color="red", linestyle="-", label=f"Low Mean: {low_mean:.2f}")

# Finalize Plot
plt.title("Confidence Interval Comparison: Top vs. Low Vendors (Profit Margin)")
plt.xlabel("Profit Margin (%)")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.show()

The confidence interval for low-performing vendors (40.48% to 42.62%) is significantly higher than that of top-performing vendors (30.74% to 31.61%).
This suggests that vendors with lower sales tend to maintain higher profit margins, potentially due to premium pricing or lower operational costs.
For High-Performing Vendors: If they aim to improve profitability, they could explore selective price adjustments, cost optimization, or bundling strategies.
For Low-Performing Vendors: Despite higher margins, their low sales volume might indicate a need for better marketing, competitive pricing, or improved distribution strategies.
# Is there a significant difference in profit margins between top-performing and low-performing vendors?
Hypothesis:
H₀ (Null Hypothesis): There is no significant difference in the mean profit margins of top-performing and low-performing vendors.
H₁ (Alternative Hypothesis): The mean profit margins of top-performing and low-performing vendors are significantly different.

# from scipy.stats import ttest_ind
# Determine thresholds for top and low-performing vendors
top_threshold = df["TotalSalesDollars"].quantile(0.75)
low_threshold = df["TotalSalesDollars"].quantile(0.25)

# Separate top and low vendors based on TotalSalesDollars
top_vendors = df[df["TotalSalesDollars"] >= top_threshold]["ProfitMargin"].dropna()
low_vendors = df[df["TotalSalesDollars"] <= low_threshold]["ProfitMargin"].dropna()

# Perform Two-Sample T-Test
t_stat, p_value = ttest_ind(top_vendors, low_vendors, equal_var=False)

# Print results
print(f"T-Statistic: {t_stat:.4f}, P-Value: {p_value:.4f}")

if p_value < 0.05:
    print("Reject H₀: There is a significant difference in profit margins between top and low-performing vendors.")
else:
    print("Fail to Reject H₀: No significant difference in profit margins.")
