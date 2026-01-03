# Raw Data Directory

This directory contains raw data from various sources.

## Structure

- `apple_watch_health/` - Apple Watch / Apple Health export data
  - Place your `export.xml` file here (exported from iPhone)
  
- `oura/` - Oura Ring data (synced via API)
  - Automatically populated by `src/data_collection/oura_api.py`
  
- `training_logs/` - Training session logs
  - Created by `src/data_collection/training_log.py`

## Privacy Note

⚠️ **All data in this directory is personal and private.**
- This directory is excluded from git (see .gitignore)
- Do NOT commit personal health data
- Keep your data secure

