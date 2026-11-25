# City-Hotel-Booking-Demand-ML-Model
GitHub: https://github.com/nourMelegy/City-Hotel-Booking-Demand-ML-Model

## Setup
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

## Classification: predicting IsCanceled

### Goal and prediction point
We predict whether a booking will be canceled. We assume the prediction is
made **when the booking is created**. So we only use information the hotel
knows at that moment.

### Leakage decisions
Dropped because they are only known later (direct or indirect leakage):
- `ReservationStatus`, `ReservationStatusDate`: they contain the answer.
- `AssignedRoomType`, `BookingChanges`: they change after booking.
- `ADR`, `DaysInWaitingList`, `RequiredCarParkingSpaces`: only known later or not needed.
- `Agent_te`: the target-encoded agent column was computed on the whole
  train file before cross-validation, so it leaks the labels. We instead keep
  the raw `Agent` ID and target-encode it **inside** the pipeline, so it is
  re-fitted on the training part of every fold.

### Data and testing plan
- Files: `data/classificationdata/H2_classification_train_with_agent.csv`
  (43,332 rows) and `data/classificationdata/H2_classification_test_with_agent.csv` (12,343 rows).
- Random seed: 42.
- All model selection and tuning use **5-fold stratified cross-validation on
  the train file only** (screening used 5 folds x 2 repeats).
- The test file is used only for the final evaluation.
- Main selection metric: **PR-AUC**, because the classes are not balanced
  (about 30% canceled). We also report accuracy, precision, recall, F1,
  ROC-AUC and the confusion matrix.

### Models
1. **Screening (untuned):** GaussianNB, Logistic Regression, KNN, Decision
   Tree, Random Forest, XGBoost, MLP, SVM (RBF kernel).
2. **Selected (best 3, by CV PR-AUC):** Random Forest, XGBoost, MLP.
3. **Tuning:** RandomizedSearchCV, scored by PR-AUC, search spaces saved in
   `outputs/classification_best_params.json`.
4. **Extra (outside the required list):** ExtraTrees, HistGradientBoosting,
   LightGBM, CatBoost, and a voting ensemble.

### Threshold
The default threshold 0.5 misses many cancellations. We tested thresholds
from 0.10 to 0.90 using out-of-fold predictions on the train file only.
Deployed model: **Random Forest (200 trees, max_depth 20, min_samples_leaf 1)**
with threshold **[0.27]**.

### PCA and LDA
PCA (50 components keeps most of the information) and LDA (1 component)
were compared with the original features for all 3 models. The original
features performed best, so the final model does not use PCA or LDA.

### Results
See `outputs/`:
- `classification_Cross-validation_Results_AfterAgent.csv`: baseline screening
- `classification_baseline_vs_tuned_cv.csv`: baseline vs tuned (CV)
- `classification_test_results.csv`: final test results + confusion values
- `classification_pca_lda_comparison_cv.csv`, `classification_pca_lda_test_results.csv`
- figures: threshold curves, confusion matrices, PCA variance plots

### Run commands
```bash
pip install -r requirements.txt

python main.py tune     --task classification --model random_forest
python main.py train    --task classification --model random_forest
python main.py evaluate --task classification --model all
python main.py results  --task classification
python main.py predict  --task classification --json classification_Test1.json
```
Trained models are saved in `artifacts/`. They are **not** in the ZIP
(project rule). 
Run `tune` and `train` once to create them before `predict`.


## Team and  MS 1 contributions

| Member | Part |
|---|---|
| Natalie David | Classification Model |
| Doaa Shaker | Regression Model |
| Laila Tarek | Regression Preprocessing |
| Nour Mostafa | Regression Preprocessing |
| Heba Hassan | Clustering |
| Malak Moustafa | Clustering Preprocessing |
