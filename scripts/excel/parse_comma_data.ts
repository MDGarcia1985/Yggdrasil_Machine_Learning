/*
 * Reconstruct and split comma-delimited structured data in Excel.
 *
 * Purpose:
 * - Recover rows that were imported into Excel as one combined text value.
 *
 * Workflow:
 * 1) Read used cells from the active worksheet.
 * 2) Reconstruct each row as one concatenated string.
 * 3) Split the row by comma and write values back into columns.
 */
function main(workbook: ExcelScript.Workbook) {
  const sheet = workbook.getActiveWorksheet();
  const usedRange = sheet.getUsedRange();

  if (!usedRange) return; // No populated cells to process.

  const values = usedRange.getValues();
  const rowCount = values.length;
  const colCount = values[0].length;

  for (let r = 0; r < rowCount; r++) {
    let combined = "";

    for (let c = 0; c < colCount; c++) {
      const val = values[r][c];
      if (val) {
        combined += val.toString();
      }
    }

    if (combined !== "") {
      const parts = combined.split(",");

      for (let c = 0; c < colCount; c++) {
        sheet.getCell(r, c).clear();
      }

      for (let c = 0; c < parts.length; c++) {
        sheet.getCell(r, c).setValue(parts[c].trim());
      }
    }
  }
}
