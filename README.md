# Predicting 30-Day Hospital Readmission Among Patients With Diabetes

## Group Members

- Harrison Furman
- Thomas Raulerson
- Serban Negoita

## Project Overview

This project examines how accurately patient demographics, hospitalization history, and clinical characteristics can predict hospital readmission within 30 days among patients with diabetes.

Using Python and Marimo, we compare three classification models to determine which provides the strongest predictive performance and evaluate whether the results could help inform discharge planning and follow-up care.

## Research Question

How accurately can patient demographics, hospitalization history, and clinical characteristics predict 30-day hospital readmission among patients with diabetes?

## Dataset

The project uses the [Diabetes 130-US Hospitals dataset](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) from the UCI Machine Learning Repository.

The dataset contains hospital encounters from 1999 to 2008.

- **Original dataset:** 101,766 hospital encounters
- **Eligible encounters:** 99,343 after excluding encounters ending in death or hospice care
- **Response variable:** 30-day hospital readmission (1 = readmitted within 30 days, 0 = not readmitted within 30 days)
- **Predictors:** 14 variables representing patient demographics, hospitalization history, and clinical characteristics

The predictors include age, gender, race, previous inpatient visits, previous emergency visits, previous outpatient visits, time in hospital, laboratory procedures, medical procedures, medications, diagnoses, HbA1c results, insulin use, and medication changes.

## Analysis

The analysis was conducted using Python in a Marimo notebook.

Three classification models were developed and evaluated:

1. Logistic Regression
2. Random Forest
3. Gradient Boosting

The analysis includes:

- Data cleaning and preprocessing
- Selection of predictors and the binary response variable
- Patient-grouped training and testing split to prevent patient overlap
- Training and evaluation of three classification models
- Confusion matrices, accuracy, sensitivity, and specificity
- ROC curves and AUROC
- Predictions for a hypothetical new patient
- Five-fold, patient-grouped cross-validation
- Model comparison and selection
- Classification threshold evaluation and justification

## Model Comparison

Five-fold, patient-grouped cross-validation was performed using the training dataset to compare model discrimination.

| Model | Mean Cross-Validation AUROC |
|---|---|
| Random Forest | 0.6412 |
| Gradient Boosting | 0.6395 |
| Logistic Regression | 0.6367 |

Random Forest achieved the highest mean cross-validation AUROC and was selected as the preferred classification model.

On the independent testing dataset, Random Forest achieved an AUROC of 0.6352, compared with 0.6329 for Gradient Boosting and 0.6289 for Logistic Regression.

Although Random Forest performed best, the differences between models were small.

## Classification Threshold Selection

At the default classification threshold of 0.50, the models demonstrated very low sensitivity.

A lower threshold of 0.12 was selected for Random Forest using five-fold, patient-grouped cross-validation on the training dataset.

This threshold was chosen to prioritize identifying approximately half of actual readmissions while retaining reasonable specificity.

## Final Results

Using the selected classification threshold of 0.12, Random Forest achieved the following results on the independent testing dataset:

| Metric | Result |
|---|---|
| Accuracy | 67.82% |
| Sensitivity | 47.56% |
| Specificity | 70.37% |
| AUROC | 0.6352 |

The model correctly identified 1,054 of the 2,216 actual 30-day readmissions but also produced 5,211 false positives.

## Conclusion

Patient demographics, hospitalization history, and clinical characteristics provide some ability to predict 30-day hospital readmission, but overall predictive performance remains modest.

Random Forest performed slightly better than Logistic Regression and Gradient Boosting. However, its limited sensitivity and substantial number of false positives mean it should not be used independently for clinical decision-making.

The model may have potential as a supplementary tool to help identify patients who could benefit from additional discharge planning or follow-up.

## Limitations

- The dataset contains hospital encounters from 1999 to 2008, which may limit applicability to current healthcare settings.
- Some clinical information was unavailable or missing, including HbA1c results.
- All three models demonstrated modest predictive discrimination.
- The selected threshold involves a tradeoff between detecting readmissions and incorrectly flagging patients.
- External validation would be necessary before considering clinical implementation.

## Accessing the Marimo Notebook

The complete analysis is available in the [`diabetes_readmission_analysis.py`](diabetes_readmission_analysis.py) file in this repository.

To run the notebook locally:

1. Clone or download this GitHub repository:

   ```bash
   git clone https://github.com/HarrisonF7/diabetes-readmission-marimo-app.git
   ```

2. Open the project folder:

   ```bash
   cd diabetes-readmission-marimo-app
   ```

3. Install the required Python packages:

   ```bash
   pip install marimo pandas scikit-learn matplotlib
   ```

4. Download the dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008).

   Place the `diabetic_data.csv` file in the project folder alongside the Marimo notebook.

5. Launch the notebook:

   ```bash
   marimo edit diabetes_readmission_analysis.py
   ```

The notebook will open in your browser, where you can run the analysis and review the model results.

## Interactive GitHub Pages App

The interactive application presents our research question, dataset, analysis process, classification model comparisons, threshold selection, and final results.

Users can explore model performance, adjust classification thresholds, examine confusion matrices, and compare predictions for a hypothetical patient.

Launch the Interactive App

## Technologies Used

- Python
- Marimo
- pandas
- scikit-learn
- matplotlib
- Git and GitHub
- GitHub Pages

## GitHub Repository

[Diabetes Readmission Prediction Project](https://github.com/HarrisonF7/diabetes-readmission-marimo-app)
