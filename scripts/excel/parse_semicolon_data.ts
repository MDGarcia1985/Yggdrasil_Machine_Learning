/*
 * File: parse_semicolon_data.ts
 * Purpose: Reconstruct and split semicolon-delimited aircraft data in Excel
 * Project: WWII Aircraft Classifier
 * Author: Michael Garcia
 * Date: 03/23/2026
 * For: CSC373 Machine Learning
 * License: MIT
*/
function main(workbook: ExcelScript.Workbook) {

  /*
   * Get active worksheet and used data range.
   * The used range defines the bounding box of all populated cells.
   */
  const sheet = workbook.getActiveWorksheet();
  const usedRange = sheet.getUsedRange();

  // Safety check: exit if no data is present
  if (!usedRange) return;

  /**
   * Extract all values into a 2D array:
   * values[row][column]
   */
  const values = usedRange.getValues();
  const rowCount = values.length;
  const colCount = values[0].length;

  /**
   * Iterate through each row and reconstruct data.
   */
  for (let r = 0; r < rowCount; r++) {

    let combined = "";

    /**
     * Step 1: Combine all non-empty cells in the row
     * into a single string.
     *
     * This handles cases where Excel split a semicolon-delimited
     * string across multiple cells.
     */
    for (let c = 0; c < colCount; c++) {
      let val = values[r][c];
      if (val) {
        combined += val.toString();
      }
    }

    /**
     * Step 2: If row contains data, split by semicolon
     * and rewrite into structured columns.
     */
    if (combined !== "") {

      let parts = combined.split(";");

      /**
       * Clear the original row before writing cleaned data.
       * This prevents leftover corrupted values.
       */
      for (let c = 0; c < colCount; c++) {
        sheet.getCell(r, c).clear();
      }

      /**
       * Write parsed values back into columns.
       */
      for (let c = 0; c < parts.length; c++) {
        sheet.getCell(r, c).setValue(parts[c].trim());
      }
    }
  }
}