# Excel Preprocessing - Semicolon Data Parser

## Overview

This Office Script fixes malformed dataset rows in Excel when aircraft data is:

- stored as a semicolon-delimited string in a single cell
- unintentionally split across multiple cells during import

It reconstructs each row into a clean, structured format suitable for machine learning preprocessing.

## Problem

The raw dataset (`ww2aircraft.csv`) may be parsed incorrectly by Excel, resulting in:

- entire records placed in a single cell
- records fragmented across multiple cells
- inconsistent column alignment

This breaks downstream processing and prevents reliable ingestion into pandas.

## Solution

The script performs two key operations:

1. **Row reconstruction**  
   Combines all non-empty cells in a row into a single string.
2. **Delimiter parsing**  
   Splits the reconstructed string using `;` and writes each value into its correct column position.

## File Location

```
scripts/
`-- excel/
    `-- parse_semicolon_data.ts
```

## When to Use This

Run this script immediately after importing the raw dataset into Excel and before exporting to CSV.

Typical workflow:

```text
Raw CSV
  -> Excel import (broken structure)
  -> Run Office Script
  -> Clean structured table
  -> Export CSV
  -> Python ML pipeline
```

## How to Use

1. Open the dataset in Excel Online or Excel Desktop.
2. If you are using Excel Desktop, save the workbook as `.xlsm` before running Office Scripts.
3. Go to the `Automate` tab.
4. Open `Code Editor`.
5. Paste the contents of `parse_semicolon_data.ts`.
6. Run the script.

## Input

Worksheet containing raw aircraft data. The data may be:

- in column A only
- scattered across multiple columns

## Output

Rows rewritten into a clean, column-aligned format, ready for:

- CSV export
- ingestion into pandas via `preprocess.py`

## Notes

- The script overwrites existing row data after reconstruction.
- Keep a backup of the raw data in `/data/raw/`.
- This step is required only once per dataset import.

## Role in Pipeline

This script is part of the data ingestion stage:

```
/data/raw -> [Excel Script] -> /data/processed -> Python pipeline
```

It ensures structural integrity before any machine learning preprocessing begins.

## Author

Michael Garcia  
WWII Aircraft Classifier Project

## License

MIT
