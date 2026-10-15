"""Manual preparation entry point; importing this module performs no IO."""
from project_config import ROOT, store_metadata_path, clickhouse_settings


def main():
    import geopandas as gpd
    import pandas as pd
    import os

    # ==========================================================
    # 1. FILE SETTINGS
    # ==========================================================

    kmz_file = str(ROOT / "Data" / "raw" / "GS25 260903 7.kmz")

    output_folder = str(ROOT / "Data" / "cleaned" / "GS25")
    os.makedirs(output_folder, exist_ok=True)

    output_file = os.path.join(
        output_folder,
        "GS25_store_coordinates.csv"
    )

    # ==========================================================
    # 2. GET ALL LAYERS
    # ==========================================================

    layers = gpd.list_layers(kmz_file)

    print("\nAvailable layers:")
    print(layers)

    # ==========================================================
    # 3. READ ALL LAYERS
    # ==========================================================

    all_data = []

    for layer_name in layers["name"]:

        print(f"\nReading: {layer_name}")

        gdf = gpd.read_file(
            kmz_file,
            layer=layer_name,
            driver="LIBKML"
        )

        # ------------------------------------------------------
        # Extract longitude and latitude from geometry
        # ------------------------------------------------------

        gdf["longitude"] = gdf.geometry.x
        gdf["latitude"] = gdf.geometry.y

        # ------------------------------------------------------
        # Keep only required columns
        # ------------------------------------------------------

        temp = gdf[
            [
                "Name",
                "latitude",
                "longitude"
            ]
        ].copy()

        # Rename store name
        temp = temp.rename(
            columns={
                "Name": "store_name"
            }
        )

        # Add source layer
        temp["source_layer"] = layer_name

        all_data.append(temp)

        print(f"Rows: {len(temp):,}")

    # ==========================================================
    # 4. COMBINE ALL LAYERS
    # ==========================================================

    df = pd.concat(
        all_data,
        ignore_index=True
    )

    # ==========================================================
    # 5. CLEAN DATA
    # ==========================================================

    df["store_name"] = (
        df["store_name"]
        .astype("string")
        .str.strip()
    )

    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce"
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce"
    )

    # Remove rows without coordinates
    df = df.dropna(
        subset=["latitude", "longitude"]
    )

    # ==========================================================
    # 6. REMOVE DUPLICATES
    # ==========================================================

    df = df.drop_duplicates(
        subset=[
            "store_name",
            "latitude",
            "longitude"
        ]
    )

    # ==========================================================
    # 7. SORT
    # ==========================================================

    df = df.sort_values(
        by=["store_name"]
    ).reset_index(drop=True)

    # ==========================================================
    # 8. SAVE CSV
    # ==========================================================

    df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    # ==========================================================
    # 9. SUMMARY
    # ==========================================================

    print("\n" + "=" * 60)
    print("FINISHED")
    print("=" * 60)

    print(f"Total stores: {len(df):,}")
    print(f"Output file: {output_file}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 10 rows:")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
