import csv
import os
import sys

# Resolve PROJECT_ROOT as the parent of the scripts directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# Define old and new path fragments
OLD_PATH_FRAGMENT = "/artifacts/./data/rendered/"
NEW_PATH_FRAGMENT = "/artifacts/rendered/"


def update_manifest_csv(csv_path, dry_run=False):
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        print(f"⚠️  Skipping empty file: {csv_path}")
        return {
            "file": csv_path,
            "updated_rows": 0,
            "total_rows": 0,
            "file_changed": False,
            "missing_files": 0,
        }

    header = rows[0]
    if "image_path" not in header:
        print(f"⚠️  No 'image_path' column in: {csv_path}")
        return {
            "file": csv_path,
            "updated_rows": 0,
            "total_rows": len(rows) - 1,
            "file_changed": False,
            "missing_files": 0,
        }

    image_path_idx = header.index("image_path")
    updated_rows = []
    updated_rows.append(header)

    updated_count = 0
    missing_file_count = 0

    for row in rows[1:]:
        original_path = row[image_path_idx]
        new_path = original_path
        if OLD_PATH_FRAGMENT in original_path:
            new_path = original_path.replace(OLD_PATH_FRAGMENT, NEW_PATH_FRAGMENT)
            updated_count += 1

        # Check existence
        if not os.path.exists(new_path):
            print(f"❌ Missing file: {new_path}")
            missing_file_count += 1

        row[image_path_idx] = new_path
        updated_rows.append(row)

    if updated_count > 0:
        if not dry_run:
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(updated_rows)
        action = "Would update" if dry_run else "✅ Updated"
        print(f"{action} {updated_count} paths in: {csv_path}")
        file_changed = True
    else:
        print(f"ℹ️ No changes needed in: {csv_path}")
        file_changed = False

    return {
        "file": csv_path,
        "updated_rows": updated_count,
        "total_rows": len(rows) - 1,
        "file_changed": file_changed,
        "missing_files": missing_file_count,
    }


def main():
    dry_run = "--dry-run" in sys.argv

    print("🔍 Searching for manifest.csv files...")
    if dry_run:
        print("🟡 Dry-run mode enabled. No files will be modified.\n")

    total_files = 0
    total_files_changed = 0
    total_rows_processed = 0
    total_rows_updated = 0
    total_missing_files = 0

    for root, dirs, files in os.walk(PROJECT_ROOT):
        for filename in files:
            if filename == "manifest.csv":
                csv_path = os.path.join(root, filename)
                result = update_manifest_csv(csv_path, dry_run=dry_run)
                total_files += 1
                total_rows_processed += result["total_rows"]
                total_rows_updated += result["updated_rows"]
                total_missing_files += result["missing_files"]
                if result["file_changed"]:
                    total_files_changed += 1

    print("\n📊 Summary")
    print(f"------------------------------------------")
    print(f" Total manifest.csv files found: {total_files}")
    print(f" Files needing changes:         {total_files_changed}")
    print(f" Total rows processed:          {total_rows_processed}")
    print(f" Total rows updated:            {total_rows_updated}")
    print(f" Rows with missing files:       {total_missing_files}")
    print(f" Dry-run mode:                  {'Yes' if dry_run else 'No'}")
    print("------------------------------------------")

    if total_missing_files > 0:
        print("⚠️  Some image files were missing! Please review the log above.")


if __name__ == "__main__":
    main()
