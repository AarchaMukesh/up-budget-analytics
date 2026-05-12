from src.extraction.parser import extract_with_pdfplumber
import pandas as pd

# Run extraction
tables = extract_with_pdfplumber("data/raw/2024-25/Gr62_Superannuation allowances and pensions.pdf")

# Combine all extracted tables into one master DataFrame for Gr62
if tables:
    df_gr62 = pd.concat(tables, ignore_index=True)
    # Save a sample to check the results
    df_gr62.head(20).to_csv("data/interim/pilot_gr62_sample.csv", index=False)
    print(f"Extracted {len(df_gr62)} rows from Gr62.")