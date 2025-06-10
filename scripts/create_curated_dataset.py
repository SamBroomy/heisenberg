#!/usr/bin/env python3
"""
Script to create a curated dataset for shipping with Heisenberg package.

This extracts a subset of GeoNames data that covers major global locations
while keeping the package size reasonable (~2-5MB).
"""

import pandas as pd
import sys
from pathlib import Path

# Feature codes to include (covers most common location types)
ESSENTIAL_FEATURE_CODES = {
    # Countries and administrative divisions
    'A.PCLI', 'A.ADM1', 'A.ADM2', 'A.ADM3',
    # Major populated places
    'P.PPLC', 'P.PPLA', 'P.PPLA2', 'P.PPLA3', 'P.PPLA4',
    # Important cities
    'P.PPL',
    # Major geographic features
    'T.MT', 'T.ISL', 'H.LK', 'H.BAY', 'S.AIRP'
}

def load_geonames_data(data_path: Path):
    """Load the full GeoNames datasets."""
    print("Loading GeoNames data...")
    
    # Column names for allCountries.txt
    columns = [
        'geonameid', 'name', 'asciiname', 'alternatenames',
        'latitude', 'longitude', 'feature_class', 'feature_code',
        'country_code', 'cc2', 'admin1_code', 'admin2_code',
        'admin3_code', 'admin4_code', 'population', 'elevation',
        'dem', 'timezone', 'modification_date'
    ]
    
    # Load main dataset
    all_countries = pd.read_csv(
        data_path / 'allCountries.txt',
        sep='\t',
        names=columns,
        low_memory=False,
        dtype={
            'geonameid': 'int32',
            'latitude': 'float32', 
            'longitude': 'float32',
            'population': 'int32',
            'elevation': 'Int32',  # Nullable integer
            'dem': 'int16'
        }
    )
    
    # Load country info
    country_info = pd.read_csv(
        data_path / 'countryInfo.txt',
        sep='\t',
        comment='#',
        names=[
            'iso', 'iso3', 'iso_numeric', 'fips', 'country',
            'capital', 'area', 'population', 'continent', 'tld',
            'currency_code', 'currency_name', 'phone', 'postal_code_format',
            'postal_code_regex', 'languages', 'geonameid', 'neighbours',
            'equivalent_fips_code'
        ]
    )
    
    # Load feature codes  
    feature_codes = pd.read_csv(
        data_path / 'featureCodes_en.txt',
        sep='\t',
        names=['code', 'name', 'description']
    )
    
    return all_countries, country_info, feature_codes

def create_curated_dataset(all_countries: pd.DataFrame, country_info: pd.DataFrame, feature_codes: pd.DataFrame):
    """Create curated dataset with essential global locations."""
    print("Creating curated dataset...")
    
    curated_data = []
    
    # 1. All countries (A.PCLI)
    print("  - Including all countries...")
    countries = all_countries[all_countries['feature_code'] == 'PCLI'].copy()
    curated_data.append(countries)
    print(f"    Added {len(countries)} countries")
    
    # 2. Major administrative divisions for key countries
    print("  - Including major administrative divisions...")
    major_countries = ['US', 'CA', 'GB', 'FR', 'DE', 'IT', 'ES', 'AU', 'JP', 'CN', 'IN', 'BR', 'RU']
    admin_divs = all_countries[
        (all_countries['country_code'].isin(major_countries)) &
        (all_countries['feature_code'].isin(['ADM1', 'ADM2']))
    ].copy()
    curated_data.append(admin_divs)
    print(f"    Added {len(admin_divs)} administrative divisions")
    
    # 3. All national capitals
    print("  - Including all national capitals...")
    capitals = all_countries[all_countries['feature_code'] == 'PPLC'].copy()
    curated_data.append(capitals)
    print(f"    Added {len(capitals)} national capitals")
    
    # 4. Major cities (population > 100k OR admin centers)
    print("  - Including major cities...")
    major_cities = all_countries[
        (
            (all_countries['population'] > 100000) |
            (all_countries['feature_code'].isin(['PPLA', 'PPLA2', 'PPLA3']))
        ) &
        (all_countries['feature_class'] == 'P')
    ].copy()
    curated_data.append(major_cities)
    print(f"    Added {len(major_cities)} major cities")
    
    # 5. Important geographic features
    print("  - Including important geographic features...")
    important_features = all_countries[
        (all_countries['feature_code'].isin(['MT', 'ISL', 'LK', 'AIRP'])) &
        (
            (all_countries['population'] > 0) |  # Has some significance
            (all_countries['elevation'] > 1000)   # Major mountains
        )
    ].copy()
    curated_data.append(important_features)
    print(f"    Added {len(important_features)} geographic features")
    
    # Combine and deduplicate
    curated = pd.concat(curated_data, ignore_index=True)
    curated = curated.drop_duplicates(subset=['geonameid']).reset_index(drop=True)
    
    print(f"Total curated entries: {len(curated)}")
    
    # Optimize data types to reduce size
    curated = optimize_datatypes(curated)
    
    return curated, country_info, feature_codes

def optimize_datatypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize data types to reduce memory usage."""
    print("  - Optimizing data types...")
    
    # Convert string columns to categories where beneficial
    category_columns = ['feature_class', 'feature_code', 'country_code', 'timezone']
    for col in category_columns:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    # Optimize numeric columns
    df['latitude'] = df['latitude'].astype('float32')
    df['longitude'] = df['longitude'].astype('float32')
    df['geonameid'] = df['geonameid'].astype('int32')
    df['population'] = df['population'].astype('int32')
    df['dem'] = df['dem'].astype('int16')
    
    return df

def save_curated_dataset(curated: pd.DataFrame, country_info: pd.DataFrame, 
                        feature_codes: pd.DataFrame, output_path: Path):
    """Save the curated dataset in optimized format."""
    print("Saving curated dataset...")
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as Parquet for optimal compression and speed
    curated.to_parquet(output_path / 'curated_locations.parquet', compression='snappy')
    country_info.to_parquet(output_path / 'country_info.parquet', compression='snappy')
    feature_codes.to_parquet(output_path / 'feature_codes.parquet', compression='snappy')
    
    # Also save as TSV for compatibility with current Rust code
    curated.to_csv(output_path / 'curated_locations.txt', sep='\t', index=False, header=False)
    country_info.to_csv(output_path / 'country_info.txt', sep='\t', index=False, header=False)
    feature_codes.to_csv(output_path / 'feature_codes.txt', sep='\t', index=False, header=False)
    
    # Print size information
    parquet_size = sum((output_path / f).stat().st_size for f in output_path.glob('*.parquet'))
    txt_size = sum((output_path / f).stat().st_size for f in output_path.glob('*.txt'))
    
    print(f"Saved curated dataset:")
    print(f"  - Parquet format: {parquet_size / 1024 / 1024:.1f} MB")
    print(f"  - Text format: {txt_size / 1024 / 1024:.1f} MB")
    print(f"  - Entries: {len(curated):,}")
    
def main():
    if len(sys.argv) != 3:
        print("Usage: python create_curated_dataset.py <input_data_path> <output_path>")
        print("Example: python create_curated_dataset.py ./raw_data ./curated_data")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    
    if not input_path.exists():
        print(f"Error: Input path {input_path} does not exist")
        sys.exit(1)
    
    try:
        # Load data
        all_countries, country_info, feature_codes = load_geonames_data(input_path)
        
        # Create curated dataset
        curated, country_info, feature_codes = create_curated_dataset(
            all_countries, country_info, feature_codes
        )
        
        # Save results
        save_curated_dataset(curated, country_info, feature_codes, output_path)
        
        print("\n✅ Successfully created curated dataset!")
        print(f"📁 Output saved to: {output_path}")
        print("\n📊 Dataset Statistics:")
        print(f"  Countries: {len(curated[curated['feature_code'] == 'PCLI'])}")
        print(f"  Cities: {len(curated[curated['feature_class'] == 'P'])}")
        print(f"  Admin divisions: {len(curated[curated['feature_class'] == 'A'])}")
        print(f"  Other features: {len(curated[~curated['feature_class'].isin(['P', 'A'])])}")
        
    except Exception as e:
        print(f"❌ Error creating curated dataset: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()