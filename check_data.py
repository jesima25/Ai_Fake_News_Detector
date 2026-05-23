# =====================================
# check_data.py
# Run this FIRST before training!
# It checks if your CSV files are correct
# =====================================
import pandas as pd

print("=" * 50)
print("  CHECKING YOUR DATASET FILES")
print("=" * 50)

fake = pd.read_csv(r"C:\Temp\FakeNewsProject\data\Fake.csv")
real = pd.read_csv(r"C:\Temp\FakeNewsProject\data\True.csv")

print(f"\nFake.csv  rows : {len(fake)}")
print(f"True.csv  rows : {len(real)}")
print(f"\nFake.csv  columns : {list(fake.columns)}")
print(f"True.csv  columns : {list(real.columns)}")

print("\n── 3 titles from Fake.csv ──────────────────")
for t in fake["title"].dropna().head(3).tolist():
    print(f"  {t[:75]}")

print("\n── 3 titles from True.csv ──────────────────")
for t in real["title"].dropna().head(3).tolist():
    print(f"  {t[:75]}")

print("\n" + "=" * 50)
print("  HOW TO READ RESULTS:")
print("  If Fake.csv has titles like 'Reuters - ...'")
print("  or 'AP - ...' then your files are SWAPPED!")
print("  Fix: rename Fake.csv to temp.csv")
print("        rename True.csv to Fake.csv")
print("        rename temp.csv to True.csv")
print("  Then run train_model.py again")
print("=" * 50)