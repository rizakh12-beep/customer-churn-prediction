# Power BI Dashboard Guide

This folder contains a **Power BI Project (PBIP)** scaffold plus everything you
need to build the churn dashboard and capture the screenshot used in the README.

```
powerbi/
├── ChurnDashboard.pbip                 <- open this in Power BI Desktop
├── ChurnDashboard.SemanticModel/       <- data model + DAX measures (pre-built)
└── ChurnDashboard.Report/              <- report with one page ("Churn Overview")
```

The data source is **`../data/powerbi_churn_ready.csv`** — the scored,
prediction-ready export produced by `python src/train_model.py`.

---

## Option A — Open the PBIP scaffold (recommended)

1. Install **Power BI Desktop** (free) from the Microsoft Store, and enable
   *File → Options → Preview features → **Power BI Project (.pbip) save option***,
   then restart.
2. Double-click **`ChurnDashboard.pbip`**.
3. The semantic model imports the CSV via a Power Query step. Update the file
   path if needed: **Home → Transform data → Churn → Source**, and point
   `File.Contents(...)` at your local `data/powerbi_churn_ready.csv`. Click
   **Close & Apply**.
4. The model already includes these **DAX measures** (in the `Churn` table):
   `Total Customers`, `Churned Customers`, `Churn Rate`, `Predicted Churn Rate`,
   `Avg Churn Probability`, `Avg Tenure`, `High Risk Customers`.
5. The **"Churn Overview" page comes pre-built with 9 visuals** (4 KPI cards +
   5 charts) — they render as soon as the data loads. Just format/tidy to taste,
   then **take the screenshot** (see *Capturing the screenshot*).

> Editing project files while Power BI Desktop is open? Changes are only picked
> up after you **fully close and reopen** Power BI Desktop.

> **Requires the PBIR + TMDL preview features** (Power BI Desktop → *File →
> Options → Preview features* → enable *Store reports using enhanced metadata
> format (PBIR)* and *Store semantic model using TMDL format*), then restart.
> All JSON files in this scaffold are validated against Microsoft's official
> Fabric schemas. If your Desktop build still refuses to open it (the TMDL model
> or a theme reference can be build-specific, and it can't be tested without
> Desktop), use **Option B** below — it takes ~10 minutes and gives an identical
> result. Tip: after building via Option B, do **File → Save as → .pbip** to have
> Power BI regenerate a guaranteed-valid project you can commit.

---

## Option B — Build from scratch (reliable fallback)

1. Open Power BI Desktop → **Get Data → Text/CSV** → select
   `data/powerbi_churn_ready.csv` → **Load**.
2. Create the measures (right-click the table → **New measure**):

   ```DAX
   Total Customers      = COUNTROWS('powerbi_churn_ready')
   Churned Customers    = SUM('powerbi_churn_ready'[Actual_Churn])
   Churn Rate           = DIVIDE([Churned Customers], [Total Customers])
   Predicted Churn Rate = DIVIDE(SUM('powerbi_churn_ready'[Predicted_Churn]), [Total Customers])
   Avg Churn Probability= AVERAGE('powerbi_churn_ready'[Churn_Probability])
   Avg Tenure           = AVERAGE('powerbi_churn_ready'[Tenure])
   High Risk Customers  = CALCULATE([Total Customers], 'powerbi_churn_ready'[Risk_Band] = "High")
   ```
   Set `Churn Rate`, `Predicted Churn Rate`, `Avg Churn Probability` format to **Percentage**.

---

## Suggested dashboard layout

Reproduce the layout in **`../images/powerbi_dashboard_mockup.png`**:

| Area | Visual | Fields |
|------|--------|--------|
| Top row | 4 **Card** visuals | `Total Customers`, `Churn Rate`, `High Risk Customers`, `Avg Churn Probability` |
| Mid-left | **Clustered column** | Axis `Contract Length`, Value `Churn Rate` |
| Mid-center | **Donut** | Legend `Risk_Band`, Value `Total Customers` |
| Mid-right | **Line** | Axis `Support Calls`, Value `Churn Rate` |
| Bottom-left | **Column histogram** | Axis `Churn_Probability` (binned), Value `Total Customers` |
| Bottom-right | **Bar** | Axis `Subscription Type`, Value `Churn Rate` |
| Optional | **Slicer** | `Gender`, `Subscription Type` |

Add a title text box: **"Customer Churn Dashboard"**.

---

## Capturing the screenshot

1. With the report open, use **File → Export → Export to PDF**, or simply take a
   screenshot of the report canvas (Windows: `Win + Shift + S`).
2. Save it as **`../images/powerbi_dashboard.png`**.
3. It will automatically appear in the README's *Power BI Dashboard* section
   (the README references `images/powerbi_dashboard.png`, with the mockup shown
   as a fallback).

---

## Note on the data used

The dashboard is built on the **held-out validation split** of the training
file (unseen, in-distribution customers), so the model's predictions are
accurate and the dashboard tells a coherent churn story. See the main
[README](../README.md#a-note-on-the-two-data-files-important) for why the
provided *testing* file is not used for the dashboard.
