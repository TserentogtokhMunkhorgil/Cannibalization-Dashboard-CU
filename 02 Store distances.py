"""Manual preparation entry point; importing this module performs no IO."""
from project_config import ROOT, store_metadata_path, clickhouse_settings


def main():
    # ==========================================================
    # 1. IMPORT PACKAGES
    # ==========================================================

    import pandas as pd
    import numpy as np

    # ==========================================================
    # 2. Store Master Data
    # ==========================================================

    store_maste_path = store_metadata_path()
    df_store_master = pd.read_excel(store_maste_path)
    df_store_master.columns
    keep_cols = ['store_id', 'store_name',
           'LATITUDE', 'LONGITUDE',
          'ost_dt']

    df_store_master = df_store_master[keep_cols]
    df_store_master

    # ==========================================================
    # 3. CLEAN LATITUDE / LONGITUDE
    # ==========================================================

    def dms_to_decimal(value):
        """
        Convert coordinates such as:
            47°54'07.2
        into decimal degrees.

        Also keeps normal decimal coordinates unchanged.
        """

        if pd.isna(value):
            return np.nan

        value = str(value).strip()

        # Already decimal
        try:
            return float(value)
        except:
            pass

        # DMS format
        try:
            value = (
                value.replace('°', ' ')
                     .replace("'", ' ')
                     .replace('"', ' ')
            )

            parts = value.split()

            if len(parts) >= 3:
                degrees = float(parts[0])
                minutes = float(parts[1])
                seconds = float(parts[2])

                return (
                    degrees
                    + minutes / 60
                    + seconds / 3600
                )

        except:
            pass

        return np.nan


    df_store_master['LATITUDE'] = (
        df_store_master['LATITUDE']
        .apply(dms_to_decimal)
    )

    df_store_master['LONGITUDE'] = (
        df_store_master['LONGITUDE']
        .apply(dms_to_decimal)
    )


    df_store_master[
        ['store_id', 'store_name', 'LATITUDE', 'LONGITUDE']
    ].head(20)





    # ==========================================================
    # 4. STORE GEO DATA
    # ==========================================================

    df_stores_geo = df_store_master[
        df_store_master['LATITUDE'].notna() &
        df_store_master['LONGITUDE'].notna()
    ].copy()

    print("Total stores:", len(df_store_master))
    print("Stores with coordinates:", len(df_stores_geo))
    print("Stores without coordinates:", len(df_store_master) - len(df_stores_geo))

    print(
        df_stores_geo[
            ['LATITUDE', 'LONGITUDE']
        ].describe()
    )


    # ==========================================================
    # 5. REMOVE DUPLICATE STORE IDs
    # ==========================================================

    print(
        "Duplicate store IDs:",
        df_stores_geo['store_id'].duplicated().sum()
    )

    df_stores_geo = (
        df_stores_geo
        .drop_duplicates(
            subset='store_id',
            keep='first'
        )
        .reset_index(drop=True)
    )

    print("Stores for distance analysis:", len(df_stores_geo))




    # ==========================================================
    # 6. CLEAN COORDINATES + HAVERSINE DISTANCE MATRIX
    # ==========================================================



    # ----------------------------------------------------------
    # Convert coordinates
    # ----------------------------------------------------------

    df_stores_geo['LATITUDE'] = (
        df_stores_geo['LATITUDE']
        .apply(dms_to_decimal)
    )

    df_stores_geo['LONGITUDE'] = (
        df_stores_geo['LONGITUDE']
        .apply(dms_to_decimal)
    )


    # ----------------------------------------------------------
    # Remove invalid coordinates
    # ----------------------------------------------------------

    df_stores_geo = df_stores_geo[
        df_stores_geo['LATITUDE'].between(-90, 90) &
        df_stores_geo['LONGITUDE'].between(-180, 180)
    ].copy()


    df_stores_geo = (
        df_stores_geo
        .drop_duplicates(
            subset='store_id',
            keep='first'
        )
        .reset_index(drop=True)
    )


    print(
        "Stores available for distance analysis:",
        len(df_stores_geo)
    )

    print(
        "\nCoordinate data types:"
    )

    print(
        df_stores_geo[
            ['LATITUDE', 'LONGITUDE']
        ].dtypes
    )


    print(
        "\nCoordinate summary:"
    )

    print(
        df_stores_geo[
            ['LATITUDE', 'LONGITUDE']
        ].describe()
    )


    # ==========================================================
    # HAVERSINE FUNCTION
    # ==========================================================

    def haversine_matrix(latitudes, longitudes):

        EARTH_RADIUS = 6371000  # meters

        lat = np.radians(
            np.asarray(latitudes, dtype=float)
        )

        lon = np.radians(
            np.asarray(longitudes, dtype=float)
        )

        dlat = lat[:, None] - lat[None, :]
        dlon = lon[:, None] - lon[None, :]

        a = (
            np.sin(dlat / 2) ** 2
            +
            np.cos(lat[:, None])
            * np.cos(lat[None, :])
            * np.sin(dlon / 2) ** 2
        )

        distance = (
            2
            * EARTH_RADIUS
            * np.arcsin(
                np.sqrt(
                    np.clip(a, 0, 1)
                )
            )
        )

        return distance


    # ==========================================================
    # CALCULATE DISTANCE MATRIX
    # ==========================================================

    distance_matrix = haversine_matrix(
        df_stores_geo['LATITUDE'],
        df_stores_geo['LONGITUDE']
    )


    print(
        "\nDistance matrix shape:",
        distance_matrix.shape
    )


    print(
        "\nDistance matrix calculated successfully."
    )

    # ==========================================================
    # 7. READABLE DISTANCE MATRIX
    # ==========================================================

    distance_matrix_df = pd.DataFrame(
        distance_matrix.round(1),
        index=df_stores_geo['store_id'],
        columns=df_stores_geo['store_id']
    )

    distance_matrix_df.head()


    # ==========================================================
    # 8. CREATE LONG-FORMAT DISTANCE TABLE
    # ==========================================================

    row_idx, col_idx = np.where(
        (distance_matrix > 0) &
        (distance_matrix <= 1500)
    )

    df_store_distance = pd.DataFrame({
        'store_id_1': df_stores_geo.iloc[row_idx]['store_id'].to_numpy(),
        'store_id_2': df_stores_geo.iloc[col_idx]['store_id'].to_numpy(),
        'distance_m': distance_matrix[
            row_idx,
            col_idx
        ]
    })

    print(df_store_distance.shape)
    print(df_store_distance.head(20))

    # ==========================================================
    # 9. ADD STORE INFORMATION
    # ==========================================================

    store_info = df_stores_geo[
        [
            'store_id',
            'store_name',
            'LATITUDE',
            'LONGITUDE',
            'ost_dt'
        ]
    ].copy()


    store_info_1 = store_info.rename(columns={
        'store_id': 'store_id_1',
        'store_name': 'store_name_1',
        'LATITUDE': 'latitude_1',
        'LONGITUDE': 'longitude_1',
        'ost_dt': 'opening_date_1'
    })


    store_info_2 = store_info.rename(columns={
        'store_id': 'store_id_2',
        'store_name': 'store_name_2',
        'LATITUDE': 'latitude_2',
        'LONGITUDE': 'longitude_2',
        'ost_dt': 'opening_date_2'
    })


    df_store_distance = df_store_distance.merge(
        store_info_1,
        on='store_id_1',
        how='left'
    )


    df_store_distance = df_store_distance.merge(
        store_info_2,
        on='store_id_2',
        how='left'
    )


    # ==========================================================
    # 10. DISTANCE BANDS
    # ==========================================================

    bins = [
        0,
        100,
        250,
        500,
        750,
        1000,
        1500
    ]

    labels = [
        '0-100m',
        '100-250m',
        '250-500m',
        '500-750m',
        '750-1000m',
        '1000-1500m'
    ]

    df_store_distance['distance_band'] = pd.cut(
        df_store_distance['distance_m'],
        bins=bins,
        labels=labels,
        right=False
    )


    # ==========================================================
    # 11. EXISTING STORE → NEW STORE
    # ==========================================================

    df_store_distance = df_store_distance[
        df_store_distance['opening_date_1'].notna() &
        df_store_distance['opening_date_2'].notna() &
        (
            df_store_distance['opening_date_1']
            <
            df_store_distance['opening_date_2']
        )
    ].copy()

    print(
        "Existing → New store pairs:",
        len(df_store_distance)
    )



    # ==========================================================
    # 12. FINAL COLUMN NAMES
    # ==========================================================

    df_store_distance = df_store_distance.rename(columns={
        'store_id_1': 'existing_store_id',
        'store_name_1': 'existing_store_name',
        'latitude_1': 'existing_latitude',
        'longitude_1': 'existing_longitude',
        'opening_date_1': 'existing_opening_date',

        'store_id_2': 'new_store_id',
        'store_name_2': 'new_store_name',
        'latitude_2': 'new_latitude',
        'longitude_2': 'new_longitude',
        'opening_date_2': 'new_opening_date'
    })


    # ==========================================================
    # 13. NEAREST NEW STORE FOR EACH EXISTING STORE
    # ==========================================================

    df_nearest_new = (
        df_store_distance
        .sort_values('distance_m')
        .groupby(
            'existing_store_id',
            as_index=False
        )
        .first()
    )

    df_nearest_new[
        [
            'existing_store_id',
            'existing_store_name',
            'new_store_id',
            'new_store_name',
            'distance_m',
            'distance_band',
            'new_opening_date'
        ]
    ].head(30)


    # ==========================================================
    # 14. SAVE DISTANCE DATA
    # ==========================================================

    df_store_distance.to_excel(
        ROOT / 'store_to_store_distance.xlsx',
        index=False
    )

    df_nearest_new.to_excel(
        ROOT / 'nearest_new_store.xlsx',
        index=False
    )

    print("Distance files saved.")


if __name__ == "__main__":
    main()

