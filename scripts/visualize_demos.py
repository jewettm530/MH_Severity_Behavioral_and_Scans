from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = Path("Dataset/phenotype/demos.csv")
OUTPUT_DIR = Path("outputs/demos_visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Read data
df = pd.read_csv(DATA_PATH, low_memory=False)
# Load the participants that were actually used by the model
merged = pd.read_csv("outputs/final_behavioral_only.csv")

merged["subjectkey"] = (
    merged["subjectkey"]
    .astype(str)
    .str.replace("_", "", regex=False)
)

df["subjectkey"] = (
    df["subjectkey"]
    .astype(str)
    .str.replace("_", "", regex=False)
)
df = df[df["subjectkey"].isin(merged["subjectkey"])].copy()

print(f"Participants after filtering to ML sample: {len(df)}")

df.columns = df.columns.str.strip()


# Dataset missing-value codes
MISSING_CODES = ["999", "99", 999, 99, ""]

# Replace known missing codes with NaN
df = df.replace(MISSING_CODES, np.nan)

print("\nSite x Group counts:")
print(pd.crosstab(df["Site"], df["Group"], margins=True))

# Convert numeric columns
numeric_cols = [
    "interview_age",
    "Age",
    "height#1_1",
    "height#2_1",
    "weight",
    "grade_completed",
    "num_sibs",
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


# Extra safety cleanup for age
if "Age" in df.columns:
    df.loc[(df["Age"] < 18) | (df["Age"] > 100), "Age"] = np.nan

if "interview_age" in df.columns:
    # interview_age appears to be in months, so convert to years if useful
    df["interview_age_years"] = df["interview_age"] / 12
    df.loc[
        (df["interview_age_years"] < 18) | (df["interview_age_years"] > 100),
        "interview_age_years"
    ] = np.nan


def clean_category_series(series):
    """
    Removes missing values and makes categories easier to read.
    """
    cleaned = series.dropna().astype(str).str.strip()

    # Remove leftover missing-like strings
    cleaned = cleaned[
        ~cleaned.isin(["999", "99", "nan", "NaN", "None", ""])
    ]

    return cleaned

def save_group_sex_heatmap():
    """
    Heatmap of participant counts by Group and Sex.
    """

    required = ["Group", "sex"]

    for col in required:
        if col not in df.columns:
            print(f"Skipping heatmap: {col} not found")
            return

    temp = df[required].copy()

    temp = temp.replace(MISSING_CODES, np.nan).dropna()

    temp["Group"] = temp["Group"].astype(str).str.strip()

    temp["sex"] = (
        temp["sex"]
        .astype(str)
        .str.strip()
        .replace({
            "F": "Female",
            "M": "Male",
            "O": "Other"
        })
    )

    table = pd.crosstab(temp["Group"], temp["sex"])

    # Keep columns in a consistent order
    desired_cols = ["Female", "Male", "Other"]
    table = table.reindex(
        columns=[c for c in desired_cols if c in table.columns]
    )

    plt.figure(figsize=(6, 4))

    plt.imshow(table.values, aspect="auto")

    plt.xticks(range(len(table.columns)), table.columns)
    plt.yticks(range(len(table.index)), table.index)

    plt.xlabel("Sex")
    plt.ylabel("Group")
    plt.title("Participant Counts by Group and Sex")

    # Write the counts inside each cell
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            plt.text(
                j,
                i,
                str(table.iloc[i, j]),
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold"
            )

    plt.colorbar(label="Participants")

    plt.tight_layout()

    out_path = OUTPUT_DIR / "group_sex_heatmap.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved: {out_path}")   
    

def save_bar_counts(column, title, filename, top_n=None):
    if column not in df.columns:
        print(f"Skipping {column}: not found")
        return

    data = clean_category_series(df[column])

    if data.empty:
        print(f"Skipping {column}: no usable data")
        return

    counts = data.value_counts()

    if top_n:
        counts = counts.head(top_n)

    plt.figure(figsize=(10, 6))
    counts.plot(kind="bar")
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("Number of participants")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    out_path = OUTPUT_DIR / filename
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def save_hist(column, title, filename, bins=20, xlabel=None):
    if column not in df.columns:
        print(f"Skipping {column}: not found")
        return

    data = df[column].dropna()

    if data.empty:
        print(f"Skipping {column}: no usable numeric data")
        return

    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel if xlabel else column)
    plt.ylabel("Number of participants")
    plt.tight_layout()

    out_path = OUTPUT_DIR / filename
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


# 1. Group distribution
save_bar_counts(
    column="Group",
    title="Participant Group Distribution",
    filename="group_distribution.png"
)

# 2. Sex distribution
save_bar_counts(
    column="sex",
    title="Sex Distribution",
    filename="sex_distribution.png"
)

# 3. Site distribution
save_bar_counts(
    column="Site",
    title="Site Distribution",
    filename="site_distribution.png"
)

# 4. Primary diagnosis distribution
def group_primary_diagnosis(dx):
    if pd.isna(dx):
        return np.nan

    dx = str(dx).strip()

    if dx in ["999", "99", "", "nan", "NaN", "None"]:
        return np.nan

    dx_lower = dx.lower().strip()

    bipolar = ["bp1", "bpi", "bp2", "bpii", "cyclothymia"]
    depressive = [
        "mdd",
        "mdd (w/ mild anxious distress, melancholic features)",
        "dysthymia",
        "depression nos",
    ]
    past_depressive = [
        "past mdd",
        "past mdd (due to sud)",
        "past dysthymia",
        "past depression",
    ]
    anxiety = [
        "gad",
        "past gad",
        "social anxiety",
        "panic disorder",
        "specific phobia",
        "anxiety nos",
        "past agoraphobia with panic disorder",
    ]
    trauma = ["ptsd", "past ptsd"]
    psychotic = ["sz", "sza", "sz (ruleout sza)", "sz or sza"]
    ocd = ["ocd", "past ocd"]
    adhd = ["adhd", "mild adhd"]
    substance = [
        "past aud",
        "past mild aud",
        "past moderate aud",
        "moderate aud",
        "severe aud",
        "mild cud",
        "moderate cud",
        "past severe cud",
        "present polysub use (er)",
    ]
    eating = ["binge eating disorder", "past unspec ed"]
    pmdd = ["pmdd"]

    if dx_lower in bipolar:
        return "Bipolar disorders"
    elif dx_lower in depressive:
        return "Depressive disorders"
    elif dx_lower in past_depressive:
        return "Past depressive disorders"
    elif dx_lower in anxiety:
        return "Anxiety disorders"
    elif dx_lower in trauma:
        return "Trauma-related disorders"
    elif dx_lower in psychotic:
        return "Psychotic disorders"
    elif dx_lower in ocd:
        return "OCD-related disorders"
    elif dx_lower in adhd:
        return "ADHD"
    elif dx_lower in substance:
        return "Substance use disorders"
    elif dx_lower in eating:
        return "Eating disorders"
    elif dx_lower in pmdd:
        return "PMDD"
    else:
        return "Other / uncategorized"


def save_grouped_primary_diagnosis_distribution():
    if "Primary_Dx" not in df.columns:
        print("Skipping grouped diagnosis distribution: Primary_Dx not found")
        return

    temp = df.copy()
    temp["Diagnosis_Group"] = temp["Primary_Dx"].apply(group_primary_diagnosis)
    temp = temp.dropna(subset=["Diagnosis_Group"])

    counts = temp["Diagnosis_Group"].value_counts()

    plt.figure(figsize=(11, 6))
    counts.plot(kind="bar")

    plt.title("Primary Diagnosis Distribution by Disorder Group")
    plt.xlabel("Disorder group")
    plt.ylabel("Number of participants")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    out_path = OUTPUT_DIR / "grouped_primary_diagnosis_distribution.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved: {out_path}")

# 5. Sex by group
if "Group" in df.columns and "sex" in df.columns:
    temp = df[["Group", "sex"]].copy()
    temp["Group"] = clean_category_series(temp["Group"])
    temp["sex"] = clean_category_series(temp["sex"])

    temp = df[["Group", "sex"]].dropna()
    temp = temp[
        ~temp["Group"].astype(str).isin(["999", "99"])
        & ~temp["sex"].astype(str).isin(["999", "99"])
    ]

    crosstab = pd.crosstab(temp["Group"], temp["sex"])

    plt.figure(figsize=(10, 6))
    crosstab.plot(kind="bar", ax=plt.gca())
    plt.title("Sex Distribution by Group")
    plt.xlabel("Group")
    plt.ylabel("Number of participants")
    plt.xticks(rotation=0)
    plt.legend(title="Sex")
    plt.tight_layout()

    out_path = OUTPUT_DIR / "sex_by_group.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

# 6. Group by site
if "Group" in df.columns and "Site" in df.columns:
    temp = df[["Group", "Site"]].dropna()
    temp = temp[
        ~temp["Group"].astype(str).isin(["999", "99"])
        & ~temp["Site"].astype(str).isin(["999", "99"])
    ]

    crosstab = pd.crosstab(temp["Site"], temp["Group"])

    plt.figure(figsize=(10, 6))
    crosstab.plot(kind="bar", ax=plt.gca())
    plt.title("Clinical Group Distribution by Site")
    plt.xlabel("Site")
    plt.ylabel("Number of participants")
    plt.xticks(rotation=0)
    plt.legend(title="Group")
    plt.tight_layout()

    out_path = OUTPUT_DIR / "group_by_site.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

# 7. Age distribution using Age in years
save_hist(
    column="Age",
    title="Participant Age Distribution",
    filename="age_distribution.png",
    bins=15,
    xlabel="Age (years)"
)

# 8. Race distribution
save_bar_counts(
    column="racial",
    title="Race Distribution",
    filename="race_distribution.png",
    top_n=15
)

df["meds_yes_no"] = df["meds_yes_no"].replace({
    1: "No",
    2: "Yes",
    "1": "No",
    "2": "Yes"
})
# 9. Medication use
save_bar_counts(
    column="meds_yes_no",
    title="Medication Use",
    filename="medication_use.png"
)

df["psych_meds"] = df["psych_meds"].replace({
    1: "No",
    2: "Yes",
    "1": "No",
    "2": "Yes"
})
# 10. Psychiatric medication use
save_bar_counts(
    column="psych_meds",
    title="Psychiatric Medication Use",
    filename="psychiatric_medication_use.png"
)

save_grouped_primary_diagnosis_distribution()
save_group_sex_heatmap()

print("\nDone. Charts saved in:", OUTPUT_DIR)