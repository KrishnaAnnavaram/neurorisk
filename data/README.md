# Data

Git does not store any dataset in this repository. Put the CSV file in this folder (git ignores it),
or point `NEURORISK_DATA` at it.

## Source

| Item | Value |
|---|---|
| Name | Alzheimer's Disease Dataset |
| Author | Rabie El Kharoua (2024) |
| URL | https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset |
| License | CC BY 4.0 (read the terms on the dataset page before you use it) |
| Nature | **Synthetic** records, made for education. The table contains no real patients. |

## Download

1. Sign in to Kaggle and accept the dataset terms.
2. Download the archive and extract `alzheimers_disease_data.csv`.
3. Put the file at `data/alzheimers_disease_data.csv`.
4. Run `neurorisk validate --data data/alzheimers_disease_data.csv`.

With the Kaggle CLI: `kaggle datasets download -d rabieelkharoua/alzheimers-disease-dataset -p data --unzip`.

## Expected file

One CSV file, one row per patient, 35 columns (the code checks these columns in `src/neurorisk/schema.py`):

| Group | Columns | Type and range |
|---|---|---|
| ID | `PatientID` | integer, used to group the splits |
| Demographic | `Age`, `Gender`, `Ethnicity`, `EducationLevel` | Age 18-120, Gender 0/1, Ethnicity 0-3 (nominal), EducationLevel 0-3 (ordinal) |
| Lifestyle | `BMI`, `Smoking`, `AlcoholConsumption`, `PhysicalActivity`, `DietQuality`, `SleepQuality` | numbers inside the ranges in `schema.py` |
| History | `FamilyHistoryAlzheimers`, `CardiovascularDisease`, `Diabetes`, `Depression`, `HeadInjury`, `Hypertension` | 0/1 |
| Clinical | `SystolicBP`, `DiastolicBP`, `CholesterolTotal`, `CholesterolLDL`, `CholesterolHDL`, `CholesterolTriglycerides` | mmHg and mg/dL |
| Cognitive and functional | `MMSE`, `FunctionalAssessment`, `ADL` | MMSE 0-30, others 0-10 |
| Symptoms | `MemoryComplaints`, `BehavioralProblems`, `Confusion`, `Disorientation`, `PersonalityChanges`, `DifficultyCompletingTasks`, `Forgetfulness` | 0/1 |
| Label | `Diagnosis` | 0/1 |
| Ignored | `DoctorInCharge` | constant text, never used |

## Offline data

You do not need the download for the demo or the tests. The generator makes a table with the same columns:

```bash
neurorisk synth --rows 2000 --seed 42 --out data/synthetic.csv
```
