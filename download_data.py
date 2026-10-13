"""
Female Labor Force Participation Data Download
Downloads all requested FLFP indicators individually with error tracking
"""

import wbdata
import pandas as pd
import os
import time
import requests
from datetime import datetime

# Create output directory
os.makedirs("data", exist_ok=True)

# Year range: end two years back (t-2), the latest year with broad coverage
# across indicators. Re-run each year to roll the data forward.
START_YEAR = 2000
END_YEAR = datetime.now().year - 2

# Complete FLFP indicator set
indicators = {
    # FLFP Measures
    "SL.TLF.CACT.FE.ZS": "flfp_15_64",

    # Demographics
    "SP.DYN.TFRT.IN": "fertility_rate",
    "SP.ADO.TFRT": "fertility_adolescent",
    "SP.URB.TOTL.IN.ZS": "urban_population",
    "SP.POP.DPND": "dependency_ratio",
    "SP.DYN.LE00.FE.IN": "life_exp_female",
    "SP.DYN.IMRT.IN": "infant_mortality",
    "SP.POP.TOTL": "population_total",  # Total population
    
    # Education
    "SE.SEC.ENRR.FE": "secondary_enroll_fe",
    "SE.TER.ENRR.FE": "tertiary_enroll_fe",
    # "SE.ADT.LITR.FE.ZS": "literacy_female",  # Removed: Too many missing observations
    "SE.ENR.PRSC.FM.ZS": "gender_parity_primary",
    "SE.ENR.SECO.FM.ZS": "gender_parity_secondary",
    
    # Economic Structure
    "NY.GDP.PCAP.KD": "gdp_per_capita_const",
    "NY.GDP.MKTP.KD.ZG": "gdp_growth",
    "NV.SRV.TOTL.ZS": "services_gdp",
    "NV.IND.TOTL.ZS": "industry_gdp",
    
    # Governance (WGI; RL.EST was archived from WDI in 2025)
    "GOV_WGI_RL.EST": "rule_of_law",
    
    # Labor Market
    "SL.UEM.TOTL.ZS": "unemployment_total",
    "SL.UEM.TOTL.FE.ZS": "unemployment_female",
    "SL.TLF.TOTL.IN": "labor_force_total",  # Total labor force (absolute)
}

# Indicators not in the default WDI source (2), mapped to their WB API source id
indicator_sources = {
    "GOV_WGI_RL.EST": 3,  # Worldwide Governance Indicators
}

print("COMPREHENSIVE FLFP Data Download")
print("=" * 50)
print(f"Downloading {len(indicators)} indicators individually")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

def download_single_indicator(code, name):
    """Download single indicator and return processed DataFrame"""
    try:
        print(f"  Downloading {name} ({code})")
        
        # Try the normal approach first
        df = wbdata.get_dataframe(
            indicators={code: name},
            date=(str(START_YEAR), str(END_YEAR)),
            source=indicator_sources.get(code),
            skip_cache=True
        )
        
        # Process the data
        df_clean = df.reset_index()
        if df_clean['date'].dtype == 'object':
            df_clean['year'] = df_clean['date'].astype(int)
        else:
            df_clean['year'] = pd.to_datetime(df_clean['date']).dt.year
        
        df_clean = df_clean.drop('date', axis=1)
        df_clean = df_clean.rename(columns={'country': 'country_name'})
        
        print(f"    SUCCESS: {df_clean.shape[0]:,} rows")
        return df_clean, None
        
    except Exception as e:
        # Check if this is a cache-related error (NoneType deletion)
        if "'NoneType' object does not support item deletion" in str(e):
            print(f"    Cache error detected, trying alternative approach...")
            try:
                # Alternative approach: Use wbdata.get_data instead of get_dataframe
                import requests
                
                # Direct API call as fallback
                url = f"https://api.worldbank.org/v2/countries/all/indicators/{code}"
                params = {
                    'format': 'json',
                    'date': f'{START_YEAR}:{END_YEAR}',
                    'per_page': 20000  # Large enough to get all data
                }
                if code in indicator_sources:
                    params['source'] = indicator_sources[code]
                
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, list) and len(data) >= 2:
                        records = data[1] if len(data) > 1 else []
                        
                        # Convert to DataFrame manually
                        rows = []
                        for record in records:
                            if record.get('value') is not None:  # Skip null values
                                rows.append({
                                    'country_name': record['country']['value'],
                                    'year': int(record['date']),
                                    name: record['value']
                                })
                        
                        if rows:
                            df_clean = pd.DataFrame(rows)
                            print(f"    SUCCESS (fallback): {df_clean.shape[0]:,} rows")
                            return df_clean, None
                        else:
                            raise ValueError("No valid data records found")
                    else:
                        raise ValueError("Unexpected API response format")
                else:
                    raise ValueError(f"API returned status {response.status_code}")
                    
            except Exception as fallback_error:
                error_msg = f"{name} ({code}): Cache error, fallback also failed - {str(fallback_error)}"
                print(f"    FAILED: {str(fallback_error)}")
                return None, error_msg
        else:
            # Non-cache related error
            error_msg = f"{name} ({code}): {str(e)}"
            print(f"    FAILED: {str(e)}")
            return None, error_msg

# Track downloads
successful_downloads = []
failed_indicators = []
failed_details = []

# Download each indicator
total = len(indicators)
for i, (code, name) in enumerate(indicators.items(), 1):
    print(f"[{i}/{total}] {name}")
    
    df_result, error = download_single_indicator(code, name)
    
    if df_result is not None:
        successful_downloads.append(df_result)
    else:
        failed_indicators.append(f"{name} ({code})")
        failed_details.append(error)
    
    if i < total:
        time.sleep(2)

print()
print("Download Summary:")
print(f"  Successful: {len(successful_downloads)}/{total}")
print(f"  Failed: {len(failed_indicators)}/{total}")

# Log failures
if failed_indicators:
    with open("data/failed_indicators.txt", 'w') as f:
        f.write(f"Failed Downloads - {datetime.now()}\n")
        f.write("=" * 40 + "\n")
        for indicator in failed_indicators:
            f.write(f"- {indicator}\n")
        f.write("\nDetailed Errors:\n")
        for detail in failed_details:
            f.write(f"- {detail}\n")

if not successful_downloads:
    print("ERROR: No indicators downloaded")
    exit(1)

print()
print("Merging successful downloads...")

# Merge all downloads
df_final = successful_downloads[0]
for i, df_chunk in enumerate(successful_downloads[1:], 2):
    print(f"  Merging {i}/{len(successful_downloads)}")
    df_final = df_final.merge(df_chunk, on=['country_name', 'year'], how='outer')

print(f"Combined: {df_final.shape}")

# Add metadata
print("Adding country metadata...")
try:
    countries = wbdata.get_countries()
    country_meta = []
    
    for country in countries:
        region = country.get('region', {}).get('value', '') if country.get('region') else ''
        income = country.get('incomeLevel', {}).get('value', '') if country.get('incomeLevel') else ''
        
        country_meta.append({
            'country_name': country['name'],
            'iso3c': country['id'],
            'region': region,
            'income_level': income
        })
    
    meta_df = pd.DataFrame(country_meta)
    df_final = df_final.merge(meta_df, on='country_name', how='left')
    print("  Added metadata")
    
except Exception as e:
    print(f"  Warning: {e}")
    df_final['iso3c'] = df_final['country_name']

# Filter countries - remove regional and income aggregates
print("Filtering to countries...")
initial = len(df_final)

# Comprehensive list of aggregates to exclude
aggregates_to_remove = [
    'Africa Eastern and Southern',
    'Africa Western and Central', 
    'East Asia & Pacific',
    'East Asia & Pacific (excluding high income)',
    'East Asia & Pacific (IDA & IBRD countries)',
    'Europe & Central Asia',
    'Europe & Central Asia (excluding high income)',
    'Europe & Central Asia (IDA & IBRD countries)',
    'Latin America & Caribbean',
    'Latin America & the Caribbean',
    'Latin America & Caribbean (excluding high income)',
    'Latin America & the Caribbean (IDA & IBRD countries)',
    'Middle East & North Africa',
    'Middle East & North Africa (excluding high income)',
    'Middle East & North Africa (IDA & IBRD countries)',
    'Middle East, North Africa, Afghanistan & Pakistan',
    'Middle East, North Africa, Afghanistan & Pakistan (IDA & IBRD)',
    'North America',
    'South Asia',
    'South Asia (IDA & IBRD)',
    'Sub-Saharan Africa',
    'Sub-Saharan Africa (excluding high income)',
    'Sub-Saharan Africa (IDA & IBRD countries)',
    'Arab World',
    'Caribbean small states',
    'Central Europe and the Baltics',
    'Early-demographic dividend',
    'Euro area',
    'European Union',
    'Fragile and conflict affected situations',
    'Heavily indebted poor countries (HIPC)',
    'High income',
    'IBRD only',
    'IDA & IBRD total',
    'IDA blend',
    'IDA only',
    'IDA total',
    'Late-demographic dividend',
    'Least developed countries: UN classification',
    'Low income',
    'Lower middle income',
    'Middle income',
    'Not classified',
    'OECD members',
    'Other small states',
    'Pacific island small states',
    'Post-demographic dividend',
    'Pre-demographic dividend',
    'Small states',
    'Upper middle income',
    'World',
    'Africa',
    'North Africa',
    'Sub-Saharan Africa (developing only)',
    'Sub-Saharan Africa excluding South Africa',
    'Sub-Saharan Africa excluding South Africa and Nigeria'
]

# Remove aggregates by name
df_final = df_final[~df_final['country_name'].isin(aggregates_to_remove)]

# Also remove any entries with region='Aggregates' 
if 'region' in df_final.columns:
    df_final = df_final[df_final['region'] != 'Aggregates']
    # Drop territories only covered by non-WDI sources (e.g. WGI's Jersey, Niue)
    df_final = df_final[df_final['region'].notna()]

print(f"  Removed {initial - len(df_final):,} non-country rows")

# Trim trailing years that have no FLFP observations yet
last_flfp_year = int(df_final.loc[df_final['flfp_15_64'].notna(), 'year'].max())
df_final = df_final[df_final['year'] <= last_flfp_year]
print(f"  Latest year with FLFP data: {last_flfp_year}")

# Drop countries with no FLFP observations (ILO does not model very small states)
has_flfp = df_final.groupby('country_name')['flfp_15_64'].transform(lambda s: s.notna().any())
print(f"  Removed {df_final.loc[~has_flfp, 'country_name'].nunique()} countries with no FLFP data")
df_final = df_final[has_flfp]

# Save
df_final = df_final.sort_values(['country_name', 'year'])
df_final.to_parquet("data/flfp_dataset.parquet", index=False)

# Summary
print()
print("FINAL SUMMARY:")
print(f"  Observations: {len(df_final):,}")
print(f"  Countries: {df_final['country_name'].nunique()}")
print(f"  Years: {df_final['year'].min()}-{df_final['year'].max()}")
print(f"  Indicators: {len(successful_downloads)}/{total}")

if failed_indicators:
    print(f"  Failed: {len(failed_indicators)} (see data/failed_indicators.txt)")

print()
print("Dataset saved: data/flfp_dataset.parquet")
print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Ready for analysis!")
