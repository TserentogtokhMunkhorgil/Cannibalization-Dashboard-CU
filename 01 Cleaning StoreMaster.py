"""Manual preparation entry point; importing this module performs no IO."""
from project_config import ROOT, store_metadata_path, clickhouse_settings


def main():
    # ==========================================================
    # 1. IMPORT PACKAGES
    # ==========================================================

    import clickhouse_connect
    import pandas as pd
    import numpy as np



    # ==========================================================
    # 2. CONNECT TO CLICKHOUSE
    # ==========================================================

    client = clickhouse_connect.get_client(**clickhouse_settings())




    # ==========================================================
    # 2. Store Master Data
    # ==========================================================

    store_maste_path = store_metadata_path()
    df_store_master = pd.read_excel(store_maste_path)
    df_store_master.columns
    keep_cols = ['store_id', 'store_name',
           'LATITUDE', 'LONGITUDE',
           'tt_type', 'wday_hours', 'wend_hours', 'ALCOHOL',
           'CIGARETTE', 'VAPE', 'ost_dt',
           'cls_dt', 'Store_Status', 'loc_area_new', 'FS', 'UB',
           'car_park_slot', 'chair_qty', 'shelf' ,'size', "Distance_Zone"]

    df_store_master = df_store_master[keep_cols]
    df_store_master



    # ==========================================================
    # 4. LOAD SALES DATA
    #
    # FINAL GRAIN:
    #
    # Month × Store
    #
    # ==========================================================

    query_sales = """
        SELECT
            toStartOfMonth(SalesDate) AS SalesMonth,
            toString(StoreID) AS StoreID,

            SUM(Sales_Income) AS Sales_Income_New,

            uniqExact(ReceiptNO) AS Transaction_Count,

            uniqExactIf(
                ReceiptNO,
                Sales_Income < 0
            ) AS Returned_Transaction_Count

        FROM
        (
            SELECT
                toDate(SalesDate) AS SalesDate,
                StoreID,
                ReceiptNO,

                SUM(
                    City_Tax
                    + Tax_VAT
                    + NetSale
                ) AS Sales_Income

            FROM
            (
                SELECT
                    SalesDate,
                    StoreID,
                    ReceiptNO,
                    NetSale,
                    City_Tax,
                    Tax_VAT

                FROM BI_DB.Cluster_Main_Sales

                WHERE
                    SalesDate >= '2018-01-01'
                    AND LCLSS_CD != '15'
            )

            GROUP BY
                SalesDate,
                StoreID,
                ReceiptNO
        )

        GROUP BY
            SalesMonth,
            StoreID
    """


    # ==========================================================
    # 5. LOAD DATA
    # ==========================================================

    df_sales = client.query_df(query_sales)


    # ==========================================================
    # 6. CALCULATE CUSTOMERS
    # ==========================================================

    df_sales['Customers'] = (
        df_sales['Transaction_Count']
        - df_sales['Returned_Transaction_Count'] * 2
    )


    # ==========================================================
    # 7. CALCULATE ACTIVE DAYS
    #
    # Active day = day with non-zero sales
    #
    # This is calculated separately because the main query
    # is aggregated to Month × Store.
    # ==========================================================

    query_active_days = """
        SELECT
            toStartOfMonth(SalesDate) AS SalesMonth,
            toString(StoreID) AS StoreID,

            uniqExact(toDate(SalesDate)) AS Active_Days

        FROM BI_DB.Cluster_Main_Sales

        WHERE
            SalesDate >= '2018-01-01'
            AND LCLSS_CD != '15'

        GROUP BY
            SalesMonth,
            StoreID
    """


    df_active_days = client.query_df(query_active_days)


    # ==========================================================
    # 8. MERGE ACTIVE DAYS
    # ==========================================================

    df_sales_agg = df_sales.merge(
        df_active_days,
        on=['SalesMonth', 'StoreID'],
        how='left'
    )


    # ==========================================================
    # 9. DATA TYPES
    # ==========================================================

    df_sales_agg['StoreID'] = df_sales_agg['StoreID'].astype(str)

    df_sales_agg['Sales_Income_New'] = (
        df_sales_agg['Sales_Income_New']
        .fillna(0)
        .astype(int)
    )

    df_sales_agg['Customers'] = (
        df_sales_agg['Customers']
        .fillna(0)
        .astype(int)
    )

    df_sales_agg['Active_Days'] = (
        df_sales_agg['Active_Days']
        .fillna(0)
        .astype(int)
    )


    # ==========================================================
    # 10. MONTHLY APSD
    # ==========================================================

    df_sales_agg['APSD_income'] = np.where(
        df_sales_agg['Active_Days'] > 0,
        df_sales_agg['Sales_Income_New']
        / df_sales_agg['Active_Days'],
        0
    )


    df_sales_agg['APSD_customers'] = np.where(
        df_sales_agg['Active_Days'] > 0,
        df_sales_agg['Customers']
        / df_sales_agg['Active_Days'],
        0
    )


    # ==========================================================
    # 11. BASKET COST
    # ==========================================================

    df_sales_agg['Basket_cost'] = np.where(
        df_sales_agg['Customers'] > 0,
        df_sales_agg['Sales_Income_New']
        / df_sales_agg['Customers'],
        0
    )


    # ==========================================================
    # 12. ROUND METRICS
    # ==========================================================

    df_sales_agg['APSD_income'] = (
        df_sales_agg['APSD_income']
        .round(0)
        .fillna(0)
    )

    df_sales_agg['APSD_customers'] = (
        df_sales_agg['APSD_customers']
        .round(0)
        .fillna(0)
    )

    df_sales_agg['Basket_cost'] = (
        df_sales_agg['Basket_cost']
        .round(0)
        .fillna(0)
    )


    # ==========================================================
    # 13. STORE ID
    # ==========================================================

    df_sales_agg['store_id'] = (
        df_sales_agg['StoreID']
        .astype(int)
    )


    # ==========================================================
    # 14. FINAL DATASET
    # ==========================================================

    df_sales_agg







    # ==========================================================
    # 2. Combine Store Master and Sales Data
    # ==========================================================


    df_cleaned = df_sales_agg.merge(df_store_master, on='store_id', how='left', validate='many_to_one')
    df_cleaned.head()

    df_cleaned.to_excel(ROOT / "store_master_data.xlsx", index=False)


    # Step 4 — Then start OpenStreetMap


if __name__ == "__main__":
    main()

