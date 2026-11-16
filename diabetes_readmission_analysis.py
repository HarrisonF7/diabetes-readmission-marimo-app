# GitHub repository: https://github.com/HarrisonF7/diabetes-readmission-marimo-app

import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Predicting 30-Day Hospital Readmission Among Patients With Diabetes

    ## Group Members

    - Harrison Furman
    - Thomas Raulerson
    - Serban Negoita

    ## GitHub Repository

    https://github.com/HarrisonF7/diabetes-readmission-marimo-app

    ## Research Question

    How accurately can patient demographics, hospitalization history, and clinical characteristics predict 30-day hospital readmission among patients with diabetes?

    ## Dataset

    This project uses the Diabetes 130-US Hospitals dataset from the UCI Machine Learning Repository.

    ## Analysis

    We will compare Logistic Regression, Random Forest, and Gradient Boosting to evaluate how accurately these models predict 30-day hospital readmission.
    """)
    return


@app.cell
def _():

    # Import pandas to work with the dataset.
    import pandas as pd

    # Load the diabetes dataset.
    df = pd.read_csv("diabetic_data.csv")

    # Check the number of rows and columns.
    print("Dataset shape:", df.shape)

    # Display the first five rows.
    df.head()
    return df, pd


@app.cell
def _(df):
    # Examine the readmission outcome before creating
    # our binary response variable.

    print("Readmission categories:")
    print(df["readmitted"].value_counts())

    print("\nReadmission percentages:")
    print(df["readmitted"].value_counts(normalize=True) * 100)
    return


@app.cell
def _(df):

    # Create a binary response variable for 30-day readmission.
    # 1 = Readmitted within 30 days
    # 0 = Not readmitted within 30 days

    df["readmitted_30"] = (df["readmitted"] == "<30").astype(int)

    # Check the distribution of the binary outcome.
    print("30-day readmission counts:")
    print(df["readmitted_30"].value_counts())

    print("\n30-day readmission percentages:")
    print(df["readmitted_30"].value_counts(normalize=True) * 100)
    return


@app.cell
def _(df):

    # Exclude encounters ending in death or hospice care.
    # These discharge disposition IDs represent:
    # 11 = Expired
    # 13 = Hospice at home
    # 14 = Hospice at a medical facility
    # 19, 20, 21 = Other death-related dispositions

    excluded_dispositions = [11, 13, 14, 19, 20, 21]

    df_clean = df[
        ~df["discharge_disposition_id"].isin(excluded_dispositions)
    ].copy()

    # Check how many encounters remain.
    print("Original encounters:", len(df))
    print("Excluded encounters:", len(df) - len(df_clean))
    print("Eligible encounters:", len(df_clean))

    # Check the outcome distribution after exclusions.
    print("\n30-day readmission rate:")
    print(df_clean["readmitted_30"].mean() * 100)
    return (df_clean,)


@app.cell
def _(df_clean):

    # Select 14 predictors representing demographics,
    # hospitalization history, and clinical characteristics.

    predictors = [
        "age",
        "gender",
        "race",
        "number_inpatient",
        "number_emergency",
        "number_outpatient",
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_diagnoses",
        "A1Cresult",
        "insulin",
        "change"
    ]

    # Separate predictors (X) and binary response (y).
    X = df_clean[predictors].copy()
    y = df_clean["readmitted_30"].copy()

    # Confirm dimensions and variable types.
    print("Predictor dataset shape:", X.shape)
    print("Response variable shape:", y.shape)

    print("\nPredictor data types:")
    print(X.dtypes)
    return X, y


@app.cell
def _(X, pd):

    # Check for missing values in our 14 predictors.
    # The dataset uses "?" for some missing information.

    X_missing = X.replace("?", float("nan"))

    # Count missing values for each predictor.
    missing_counts = X_missing.isna().sum()

    # Calculate the percentage of missing values.
    missing_percentages = X_missing.isna().mean() * 100

    # Display the missing data summary.
    missing_summary = pd.DataFrame({
        "Missing Count": missing_counts,
        "Missing Percentage": missing_percentages
    })

    print("Missing values by predictor:")
    print(missing_summary.to_string())
    return (X_missing,)


@app.cell
def _(X_missing):

    # Create a copy of our predictors for preprocessing.
    X_clean = X_missing.copy()

    # Preserve missing race information as a separate category.
    X_clean["race"] = X_clean["race"].fillna("Unknown")

    # Preserve missing A1C results as "Not measured".
    X_clean["A1Cresult"] = X_clean["A1Cresult"].fillna("Not measured")

    # Verify that no missing values remain.
    print("Total missing values:", X_clean.isna().sum().sum())

    # Examine the updated categories.
    print("\nRace categories:")
    print(X_clean["race"].value_counts())

    print("\nA1C result categories:")
    print(X_clean["A1Cresult"].value_counts())
    return (X_clean,)


@app.cell
def _(X_clean, df_clean, y):

    # Import GroupShuffleSplit to separate patients
    # without placing the same patient in both datasets.
    from sklearn.model_selection import GroupShuffleSplit

    # Use patient identifiers to group encounters.
    patient_groups = df_clean["patient_nbr"]

    # Create an approximately 80/20 train-test split.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42
    )

    train_idx, test_idx = next(
        splitter.split(X_clean, y, groups=patient_groups)
    )

    # Create training and testing datasets.
    X_train = X_clean.iloc[train_idx].copy()
    X_test = X_clean.iloc[test_idx].copy()

    y_train = y.iloc[train_idx].copy()
    y_test = y.iloc[test_idx].copy()

    # Confirm that patients do not overlap.
    train_patients = set(patient_groups.iloc[train_idx])
    test_patients = set(patient_groups.iloc[test_idx])

    print("Training encounters:", len(X_train))
    print("Testing encounters:", len(X_test))

    print("\nTraining readmission rate:",
          round(y_train.mean() * 100, 2), "%")

    print("Testing readmission rate:",
          round(y_test.mean() * 100, 2), "%")

    print("\nPatients appearing in both sets:",
          len(train_patients.intersection(test_patients)))
    return X_test, X_train, patient_groups, train_idx, y_test, y_train


@app.cell
def _(X_train):

    # Import preprocessing tools.
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    # Identify numerical and categorical predictors.
    numeric_features = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        include=["object", "str", "category"]
    ).columns.tolist()

    # Create a preprocessing pipeline.
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            ("categorical", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ), categorical_features)
        ]
    )

    # Display the selected variables.
    print("Numerical predictors:")
    print(numeric_features)

    print("\nCategorical predictors:")
    print(categorical_features)

    print("\nNumber of numerical predictors:", len(numeric_features))
    print("Number of categorical predictors:", len(categorical_features))
    return (preprocessor,)


@app.cell
def _(X_test, X_train, preprocessor, y_train):

    # Model 1: Logistic Regression
    # Train a baseline classification model using
    # our prepared numerical and categorical predictors.

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    # Create a pipeline combining preprocessing
    # with Logistic Regression.
    logistic_model = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ])

    # Train the model using training data only.
    logistic_model.fit(X_train, y_train)

    # Generate predicted probabilities on the test set.
    logistic_probabilities = logistic_model.predict_proba(
        X_test
    )[:, 1]

    # Classify patients using the default 0.50 threshold.
    logistic_predictions = (
        logistic_probabilities >= 0.50
    ).astype(int)

    print("Logistic Regression trained successfully!")
    print("Number of test predictions:",
          len(logistic_predictions))

    print("\nPredicted readmissions:",
          logistic_predictions.sum())

    print("Predicted non-readmissions:",
          (logistic_predictions == 0).sum())
    return (
        Pipeline,
        logistic_model,
        logistic_predictions,
        logistic_probabilities,
    )


@app.cell
def _(logistic_predictions, logistic_probabilities, y_test):

    # Evaluate Logistic Regression on the testing set.
    # Calculate the confusion matrix and classification metrics.

    from sklearn.metrics import (
        confusion_matrix,
        accuracy_score,
        recall_score,
        roc_auc_score
    )

    # Create the confusion matrix.
    logistic_cm = confusion_matrix(
        y_test, logistic_predictions
    )

    # Extract the four confusion matrix values.
    tn, fp, fn, tp = logistic_cm.ravel()

    # Calculate performance metrics.
    logistic_accuracy = accuracy_score(
        y_test, logistic_predictions
    )

    logistic_sensitivity = recall_score(
        y_test, logistic_predictions
    )

    logistic_specificity = tn / (tn + fp)

    logistic_auroc = roc_auc_score(
        y_test, logistic_probabilities
    )

    # Display the results.
    print("Logistic Regression Results")
    print("---------------------------")

    print("\nConfusion Matrix:")
    print(logistic_cm)

    print("\nAccuracy:",
          round(logistic_accuracy, 4))

    print("Sensitivity:",
          round(logistic_sensitivity, 4))

    print("Specificity:",
          round(logistic_specificity, 4))

    print("AUROC:",
          round(logistic_auroc, 4))
    return (
        accuracy_score,
        confusion_matrix,
        logistic_accuracy,
        logistic_auroc,
        logistic_sensitivity,
        logistic_specificity,
        recall_score,
        roc_auc_score,
    )


@app.cell
def _(logistic_auroc, logistic_probabilities, y_test):

    # Plot the ROC curve for Logistic Regression.
    # The ROC curve shows sensitivity versus false positive
    # rate across all possible classification thresholds.

    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    # Calculate ROC curve coordinates.
    logistic_fpr, logistic_tpr, logistic_thresholds = roc_curve(
        y_test,
        logistic_probabilities
    )

    # Create the ROC curve.
    plt.figure(figsize=(7, 5))

    plt.plot(
        logistic_fpr,
        logistic_tpr,
        label=f"Logistic Regression (AUROC = {logistic_auroc:.4f})",
        linewidth=2
    )

    # Add the reference line representing random classification.
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        label="Random Classifier (AUROC = 0.50)"
    )

    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("ROC Curve: Logistic Regression")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    plt.show()
    return plt, roc_curve


@app.cell
def _(logistic_model, pd):

    # Demonstrate Logistic Regression on a hypothetical new patient.
    # This patient is not taken from the testing dataset.

    new_patient = pd.DataFrame([{
        "age": "[60-70)",
        "gender": "Male",
        "race": "Caucasian",
        "number_inpatient": 2,
        "number_emergency": 1,
        "number_outpatient": 0,
        "time_in_hospital": 5,
        "num_lab_procedures": 45,
        "num_procedures": 1,
        "num_medications": 18,
        "number_diagnoses": 7,
        "A1Cresult": ">8",
        "insulin": "Steady",
        "change": "Ch"
    }])

    # Predict the probability of readmission.
    new_patient_probability = logistic_model.predict_proba(
        new_patient
    )[0, 1]

    # Classify the patient using the initial 0.50 threshold.
    new_patient_prediction = int(
        new_patient_probability >= 0.50
    )

    print("Logistic Regression: New Patient Prediction")
    print("-------------------------------------------")

    print("Predicted 30-day readmission probability:",
          round(new_patient_probability * 100, 2), "%")

    print("Classification:",
          "Readmitted within 30 days"
          if new_patient_prediction == 1
          else "Not readmitted within 30 days")

    print("Classification threshold: 0.50")
    return (new_patient,)


@app.cell
def _(Pipeline, X_test, X_train, preprocessor, y_train):

    # Model 2: Random Forest
    # Train an ensemble of decision trees to predict
    # 30-day hospital readmission.

    from sklearn.ensemble import RandomForestClassifier

    # Create a pipeline combining preprocessing
    # with Random Forest classification.
    random_forest_model = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=30,
            random_state=42,
            n_jobs=-1
        ))
    ])

    # Train the model using training data only.
    random_forest_model.fit(X_train, y_train)

    # Generate predicted probabilities on the testing set.
    rf_probabilities = random_forest_model.predict_proba(
        X_test
    )[:, 1]

    # Classify patients using the initial 0.50 threshold.
    rf_predictions = (
        rf_probabilities >= 0.50
    ).astype(int)

    print("Random Forest trained successfully!")
    print("Number of test predictions:",
          len(rf_predictions))

    print("\nPredicted readmissions:",
          rf_predictions.sum())

    print("Predicted non-readmissions:",
          (rf_predictions == 0).sum())
    return random_forest_model, rf_predictions, rf_probabilities


@app.cell
def _(
    accuracy_score,
    confusion_matrix,
    recall_score,
    rf_predictions,
    rf_probabilities,
    roc_auc_score,
    y_test,
):

    # Evaluate Random Forest using the testing dataset.
    # Calculate the confusion matrix, accuracy,
    # sensitivity, specificity, and AUROC.

    # Create the confusion matrix.
    rf_cm = confusion_matrix(
        y_test, rf_predictions
    )

    # Extract confusion matrix values.
    rf_tn, rf_fp, rf_fn, rf_tp = rf_cm.ravel()

    # Calculate performance metrics.
    rf_accuracy = accuracy_score(
        y_test, rf_predictions
    )

    rf_sensitivity = recall_score(
        y_test, rf_predictions
    )

    rf_specificity = rf_tn / (rf_tn + rf_fp)

    rf_auroc = roc_auc_score(
        y_test, rf_probabilities
    )

    # Display results.
    print("Random Forest Results")
    print("---------------------")

    print("\nConfusion Matrix:")
    print(rf_cm)

    print("\nAccuracy:",
          round(rf_accuracy, 4))

    print("Sensitivity:",
          round(rf_sensitivity, 4))

    print("Specificity:",
          round(rf_specificity, 4))

    print("AUROC:",
          round(rf_auroc, 4))
    return rf_accuracy, rf_auroc, rf_sensitivity, rf_specificity


@app.cell
def _(plt, rf_auroc, rf_probabilities, roc_curve, y_test):

    # Plot the ROC curve for Random Forest.
    # This evaluates model discrimination across
    # different classification thresholds.

    # Calculate ROC curve coordinates.
    rf_fpr, rf_tpr, rf_thresholds = roc_curve(
        y_test,
        rf_probabilities
    )

    # Create the ROC curve.
    plt.figure(figsize=(7, 5))

    plt.plot(
        rf_fpr,
        rf_tpr,
        label=f"Random Forest (AUROC = {rf_auroc:.4f})",
        linewidth=2
    )

    # Add the reference line for random classification.
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        label="Random Classifier (AUROC = 0.50)"
    )

    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("ROC Curve: Random Forest")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    plt.show()
    return


@app.cell
def _(new_patient, random_forest_model):

    # Demonstrate Random Forest on the same hypothetical
    # patient used for Logistic Regression.

    # Predict the probability of 30-day readmission.
    rf_patient_probability = random_forest_model.predict_proba(
        new_patient
    )[0, 1]

    # Classify the patient using the initial 0.50 threshold.
    rf_patient_prediction = int(
        rf_patient_probability >= 0.50
    )

    # Display the results.
    print("Random Forest: New Patient Prediction")
    print("-------------------------------------")

    print("Predicted 30-day readmission probability:",
          round(rf_patient_probability * 100, 2), "%")

    print("Classification:",
          "Readmitted within 30 days"
          if rf_patient_prediction == 1
          else "Not readmitted within 30 days")

    print("Classification threshold: 0.50")
    return


@app.cell
def _(Pipeline, X_test, X_train, preprocessor, y_train):

    # Model 3: Gradient Boosting
    # Train a sequence of decision trees that progressively
    # improve predictions of 30-day hospital readmission.

    from sklearn.ensemble import GradientBoostingClassifier

    # Create a pipeline combining preprocessing
    # with Gradient Boosting classification.
    gradient_boosting_model = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ))
    ])

    # Train the model using training data only.
    gradient_boosting_model.fit(X_train, y_train)

    # Generate predicted probabilities on the testing set.
    gb_probabilities = gradient_boosting_model.predict_proba(
        X_test
    )[:, 1]

    # Classify patients using the initial 0.50 threshold.
    gb_predictions = (
        gb_probabilities >= 0.50
    ).astype(int)

    print("Gradient Boosting trained successfully!")
    print("Number of test predictions:",
          len(gb_predictions))

    print("\nPredicted readmissions:",
          gb_predictions.sum())

    print("Predicted non-readmissions:",
          (gb_predictions == 0).sum())
    return gb_predictions, gb_probabilities, gradient_boosting_model


@app.cell
def _(
    accuracy_score,
    confusion_matrix,
    gb_predictions,
    gb_probabilities,
    recall_score,
    roc_auc_score,
    y_test,
):

    # Evaluate Gradient Boosting using the testing dataset.
    # Calculate the confusion matrix, accuracy,
    # sensitivity, specificity, and AUROC.

    # Create the confusion matrix.
    gb_cm = confusion_matrix(
        y_test, gb_predictions
    )

    # Extract confusion matrix values.
    gb_tn, gb_fp, gb_fn, gb_tp = gb_cm.ravel()

    # Calculate performance metrics.
    gb_accuracy = accuracy_score(
        y_test, gb_predictions
    )

    gb_sensitivity = recall_score(
        y_test, gb_predictions
    )

    gb_specificity = gb_tn / (gb_tn + gb_fp)

    gb_auroc = roc_auc_score(
        y_test, gb_probabilities
    )

    # Display results.
    print("Gradient Boosting Results")
    print("-------------------------")

    print("\nConfusion Matrix:")
    print(gb_cm)

    print("\nAccuracy:",
          round(gb_accuracy, 4))

    print("Sensitivity:",
          round(gb_sensitivity, 4))

    print("Specificity:",
          round(gb_specificity, 4))

    print("AUROC:",
          round(gb_auroc, 4))
    return gb_accuracy, gb_auroc, gb_sensitivity, gb_specificity


@app.cell
def _(gb_auroc, gb_probabilities, plt, roc_curve, y_test):

    # Plot the ROC curve for Gradient Boosting.
    # This evaluates how well the model distinguishes
    # readmissions from non-readmissions across thresholds.

    # Calculate ROC curve coordinates.
    gb_fpr, gb_tpr, gb_thresholds = roc_curve(
        y_test,
        gb_probabilities
    )

    # Create the ROC curve.
    plt.figure(figsize=(7, 5))

    plt.plot(
        gb_fpr,
        gb_tpr,
        label=f"Gradient Boosting (AUROC = {gb_auroc:.4f})",
        linewidth=2
    )

    # Add the reference line for random classification.
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        label="Random Classifier (AUROC = 0.50)"
    )

    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("ROC Curve: Gradient Boosting")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    plt.show()
    return


@app.cell
def _(gradient_boosting_model, new_patient):

    # Demonstrate Gradient Boosting on the same hypothetical
    # patient used for Logistic Regression and Random Forest.

    # Predict the probability of 30-day readmission.
    gb_patient_probability = gradient_boosting_model.predict_proba(
        new_patient
    )[0, 1]

    # Classify the patient using the initial 0.50 threshold.
    gb_patient_prediction = int(
        gb_patient_probability >= 0.50
    )

    # Display the results.
    print("Gradient Boosting: New Patient Prediction")
    print("-----------------------------------------")

    print("Predicted 30-day readmission probability:",
          round(gb_patient_probability * 100, 2), "%")

    print("Classification:",
          "Readmitted within 30 days"
          if gb_patient_prediction == 1
          else "Not readmitted within 30 days")

    print("Classification threshold: 0.50")
    return


@app.cell
def _(
    gb_accuracy,
    gb_auroc,
    gb_sensitivity,
    gb_specificity,
    logistic_accuracy,
    logistic_auroc,
    logistic_sensitivity,
    logistic_specificity,
    pd,
    rf_accuracy,
    rf_auroc,
    rf_sensitivity,
    rf_specificity,
):

    # Compare the three classification models using
    # their performance on the same testing dataset.
    # All classification metrics use the 0.50 threshold.

    model_comparison = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Random Forest",
            "Gradient Boosting"
        ],
        "Accuracy": [
            logistic_accuracy,
            rf_accuracy,
            gb_accuracy
        ],
        "Sensitivity": [
            logistic_sensitivity,
            rf_sensitivity,
            gb_sensitivity
        ],
        "Specificity": [
            logistic_specificity,
            rf_specificity,
            gb_specificity
        ],
        "AUROC": [
            logistic_auroc,
            rf_auroc,
            gb_auroc
        ]
    })

    # Sort models by AUROC, highest first.
    model_comparison = model_comparison.sort_values(
        by="AUROC",
        ascending=False
    )

    # Display the comparison table.
    print("Classification Model Comparison")
    print("--------------------------------")
    print(model_comparison.to_string(index=False))
    return


@app.cell
def _(
    X_train,
    confusion_matrix,
    patient_groups,
    pd,
    random_forest_model,
    train_idx,
    y_train,
):

    # Evaluate Random Forest thresholds using training data only.
    # GroupKFold keeps encounters from the same patient together.
    # Out-of-fold probabilities prevent threshold selection
    # from using the held-out testing dataset.

    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.base import clone

    # Identify patient groups in the training set.
    rf_train_groups = patient_groups.iloc[train_idx]

    # Create five patient-grouped cross-validation folds.
    rf_cv = GroupKFold(n_splits=5)

    # Generate out-of-fold readmission probabilities.
    rf_cv_probabilities = cross_val_predict(
        clone(random_forest_model),
        X_train,
        y_train,
        groups=rf_train_groups,
        cv=rf_cv,
        method="predict_proba",
        n_jobs=-1
    )[:, 1]

    # Evaluate candidate classification thresholds.
    rf_candidate_thresholds = [
        0.05, 0.08, 0.10, 0.12,
        0.15, 0.18, 0.20, 0.25, 0.30
    ]

    rf_threshold_results = []

    for rf_threshold in rf_candidate_thresholds:
        rf_cv_predictions = (
            rf_cv_probabilities >= rf_threshold
        ).astype(int)

        rf_cv_tn, rf_cv_fp, rf_cv_fn, rf_cv_tp = confusion_matrix(
            y_train, rf_cv_predictions
        ).ravel()

        rf_threshold_results.append({
            "Threshold": rf_threshold,
            "Sensitivity": rf_cv_tp / (rf_cv_tp + rf_cv_fn),
            "Specificity": rf_cv_tn / (rf_cv_tn + rf_cv_fp),
            "Accuracy": (
                (rf_cv_tp + rf_cv_tn) /
                (rf_cv_tp + rf_cv_tn + rf_cv_fp + rf_cv_fn)
            ),
            "Predicted Readmissions": int(rf_cv_predictions.sum())
        })

    # Display threshold comparison.
    rf_threshold_table = pd.DataFrame(rf_threshold_results)

    print("Random Forest: Training Cross-Validation Thresholds")
    print("---------------------------------------------------")
    print(rf_threshold_table.to_string(index=False))
    return rf_cv, rf_train_groups


@app.cell
def _(
    accuracy_score,
    confusion_matrix,
    recall_score,
    rf_probabilities,
    roc_auc_score,
    y_test,
):

    # Evaluate the selected Random Forest threshold
    # on the untouched testing dataset.
    # The 0.12 threshold was chosen using training
    # cross-validation results, not testing outcomes.

    rf_final_threshold = 0.12

    # Classify testing encounters using the new threshold.
    rf_final_predictions = (
        rf_probabilities >= rf_final_threshold
    ).astype(int)

    # Calculate the confusion matrix.
    rf_final_cm = confusion_matrix(
        y_test, rf_final_predictions
    )

    rf_final_tn, rf_final_fp, rf_final_fn, rf_final_tp = (
        rf_final_cm.ravel()
    )

    # Calculate final performance metrics.
    rf_final_accuracy = accuracy_score(
        y_test, rf_final_predictions
    )

    rf_final_sensitivity = recall_score(
        y_test, rf_final_predictions
    )

    rf_final_specificity = (
        rf_final_tn / (rf_final_tn + rf_final_fp)
    )

    # AUROC is unchanged because it evaluates
    # predicted probabilities across all thresholds.
    rf_final_auroc = roc_auc_score(
        y_test, rf_probabilities
    )

    # Display final results.
    print("Random Forest: Final Test Results")
    print("---------------------------------")
    print("Selected threshold:", rf_final_threshold)

    print("\nConfusion Matrix:")
    print(rf_final_cm)

    print("\nAccuracy:",
          round(rf_final_accuracy, 4))

    print("Sensitivity:",
          round(rf_final_sensitivity, 4))

    print("Specificity:",
          round(rf_final_specificity, 4))

    print("AUROC:",
          round(rf_final_auroc, 4))

    print("\nPredicted readmissions:",
          int(rf_final_predictions.sum()))
    return


@app.cell
def _():
    # Import Marimo to display formatted Markdown sections.
    import marimo as mo

    return (mo,)


@app.cell
def _(
    X_train,
    gradient_boosting_model,
    logistic_model,
    pd,
    random_forest_model,
    rf_cv,
    rf_train_groups,
    y_train,
):

    # Compare all three models using training data only.
    # GroupKFold ensures that encounters from the same
    # patient are not split between validation folds.

    from sklearn.model_selection import cross_val_score

    # Models already trained in previous cells.
    cv_models = {
        "Logistic Regression": logistic_model,
        "Random Forest": random_forest_model,
        "Gradient Boosting": gradient_boosting_model
    }

    # Store cross-validation results.
    cv_model_results = []

    for cv_model_name, cv_model_pipeline in cv_models.items():

        # Evaluate AUROC across five patient-grouped folds.
        cv_auroc_scores = cross_val_score(
            cv_model_pipeline,
            X_train,
            y_train,
            groups=rf_train_groups,
            cv=rf_cv,
            scoring="roc_auc",
            n_jobs=-1
        )

        cv_model_results.append({
            "Model": cv_model_name,
            "Mean CV AUROC": cv_auroc_scores.mean(),
            "SD CV AUROC": cv_auroc_scores.std()
        })

    # Create and sort the comparison table.
    cv_model_comparison = pd.DataFrame(cv_model_results)

    cv_model_comparison = cv_model_comparison.sort_values(
        by="Mean CV AUROC",
        ascending=False
    )

    print("Five-Fold Training Cross-Validation Comparison")
    print("----------------------------------------------")
    print(cv_model_comparison.to_string(index=False))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model Selection and Threshold Justification

    ### Comparison of Classification Models

    Three classification models were trained and evaluated to predict 30-day hospital readmission among patients with diabetes: Logistic Regression, Random Forest, and Gradient Boosting.

    Five-fold, patient-grouped cross-validation was performed using the training dataset to compare model discrimination.

    Random Forest achieved the highest mean cross-validation AUROC (0.6412), followed by Gradient Boosting (0.6395) and Logistic Regression (0.6367). The cross-validation standard deviations were similar across models, ranging from approximately 0.0120 to 0.0128.

    Based on these training-data validation results, Random Forest was selected as the preferred classification model.

    On the independent testing dataset, Random Forest achieved an AUROC of 0.6352, compared with 0.6329 for Gradient Boosting and 0.6289 for Logistic Regression.

    Although Random Forest performed best, the differences between models were small and do not establish statistical superiority.

    ### Classification Threshold Selection

    At the default classification threshold of 0.50, all three models demonstrated very low sensitivity. Random Forest identified no actual readmissions at this threshold.

    A lower threshold of 0.12 was selected for Random Forest using five-fold, patient-grouped cross-validation on the training dataset.

    At this threshold, training cross-validation produced 49.07% sensitivity and 70.18% specificity.

    The threshold was selected to prioritize identifying approximately half of actual readmissions while retaining reasonable specificity.

    This choice reflects the potential value of identifying patients who may benefit from additional discharge planning or follow-up, while acknowledging the increased number of false positives.

    ### Final Random Forest Performance

    Using the selected 0.12 threshold, Random Forest achieved the following results on the independent testing dataset:

    - Accuracy: 67.82%
    - Sensitivity: 47.56%
    - Specificity: 70.37%
    - AUROC: 0.6352

    The model correctly identified 1,054 of the 2,216 actual 30-day readmissions but also produced 5,211 false positives.

    ### Interpretation

    Random Forest was selected based on its highest mean cross-validation AUROC among the three models.

    The independent testing results supported this selection, although overall predictive discrimination remained modest.

    The model's limited sensitivity and substantial number of false positives prevent it from being appropriate for independent clinical decision-making.

    Its predictions may be more useful as supplementary information for discharge planning and follow-up review, rather than as a replacement for clinical judgment.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Final Conclusion

    ### Research Question

    How accurately can patient demographics, hospitalization history, and clinical characteristics predict 30-day hospital readmission among patients with diabetes?

    ### Main Findings

    Using 99,343 eligible hospital encounters, we developed and evaluated three classification models: Logistic Regression, Random Forest, and Gradient Boosting.

    Random Forest achieved the highest test AUROC of 0.6352, although all three models demonstrated relatively modest predictive discrimination.

    After selecting a classification threshold of 0.12 using training-data cross-validation, Random Forest identified 47.56% of actual 30-day readmissions in the testing dataset, with 70.37% specificity and 67.82% accuracy.

    ### Clinical Significance

    These findings suggest that routinely collected patient information provides some ability to distinguish between encounters with and without 30-day readmission.

    However, the model missed approximately 52% of actual readmissions and produced a substantial number of false positives.

    Therefore, the model should not be used independently to make clinical decisions. It may have potential as a supplementary tool for identifying patients who could benefit from additional discharge planning or follow-up.

    ### Limitations

    - The dataset contains hospital encounters from 1999 to 2008, which may limit applicability to current healthcare settings.
    - Some clinical variables were unavailable or had substantial missing information, including HbA1c results.
    - The models demonstrated only modest discrimination.
    - The selected threshold involves a tradeoff between detecting readmissions and incorrectly flagging patients.
    - External validation would be necessary before considering clinical implementation.

    ### Overall Conclusion

    Patient demographics, hospitalization history, and clinical characteristics can predict 30-day hospital readmission better than random chance, but predictive performance remains limited.

    Random Forest was the strongest-performing model by test AUROC in this analysis, although its advantage over Logistic Regression and Gradient Boosting was small.

    Additional predictors, model refinement, and external validation would be needed to improve clinical usefulness.
    """)
    return


if __name__ == "__main__":
    app.run()
