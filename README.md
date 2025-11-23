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
