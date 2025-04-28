import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import lightgbm as lgb

from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer, mean_squared_log_error
from scipy.sparse import hstack, csr_matrix
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

# --- Configuration & Setup ---
# Display settings for Pandas
pd.set_option('display.max_columns', None)

# Plot settings
plt.rc('font', family='Malgun Gothic') # Use Malgun Gothic font for Korean characters
plt.rcParams['axes.unicode_minus'] = False # Fix minus sign display

# --- Synthetic Data Generation ---
# Note: This function generates synthetic data by randomly sampling existing values.
# This might not preserve correlations between features. Consider more advanced methods
# like SMOGN if realistic data augmentation is crucial.
def generate_synthetic_data(df, n_samples=200):
    """Generates synthetic defect data by sampling from existing data."""
    # If 'defect_description' is available and needed, integrate its generation here.
    # descriptions = {
    #     'Cosmetic': ["표면 긁힘 발생", "색상 불균형", "마감 불량 확인"],
    #     'Functional': ["버튼 작동 불량", "센서 오작동", "전류 흐름 이상"],
    #     'Structural': ["프레임 틀어짐", "내부 연결부 손상", "하우징 크랙 발견"]
    # }

    new_data = []
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    # Ensure required columns exist for sampling
    required_cols = ['defect_type', 'defect_location', 'severity', 'inspection_method', 'product_id', 'repair_cost']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    for _ in range(n_samples):
        row = {}
        # Sample categorical features
        for col in categorical_cols:
            if col in required_cols and col != 'defect_description': # Avoid sampling description if not handled
                row[col] = random.choice(df[col].dropna().unique())

        # Handle repair_cost separately for controlled generation
        base_cost = df['repair_cost'].mean()
        std_cost = df['repair_cost'].std()
        row['repair_cost'] = float(np.clip(np.random.normal(base_cost, std_cost), 10, 1500))

        # Sample other numeric features if they exist
        for col in numeric_cols:
            if col not in ['repair_cost'] and col in df.columns: # Avoid resampling repair_cost
                # Ensure there are non-NaN values to sample from
                if not df[col].dropna().empty:
                    row[col] = random.choice(df[col].dropna().values)
                else:
                    row[col] = np.nan # Or some other default value

        # Placeholder for defect_description generation if needed
        # if 'defect_description' in required_cols:
        #    defect_type = row.get('defect_type')
        #    row['defect_description'] = random.choice(descriptions.get(defect_type, ["결함 설명 없음"])) if defect_type else "결함 설명 없음"

        new_data.append(row)

    return pd.DataFrame(new_data)

# --- Data Loading and Initial Preparation ---
def load_and_prepare_data(url, n_synthetic_samples=500, top_n_products=10):
    """Loads data, generates synthetic samples, handles outliers, and basic types."""
    print("Loading data...")
    df = pd.read_csv(url)

    # Ensure 'defect_date' exists before generating synthetic data if it's needed there
    if 'defect_date' in df.columns:
        df['defect_date'] = pd.to_datetime(df['defect_date'], errors='coerce')
    else:
        print("Warning: 'defect_date' column not found. Date features cannot be created.")
        # Add a placeholder if necessary, or handle downstream logic
        # df['defect_date'] = pd.NaT # Example placeholder

    print("Generating synthetic data...")
    # Pass only necessary columns if generate_synthetic_data expects specific ones
    # Or modify generate_synthetic_data to be more flexible
    synthetic_df = generate_synthetic_data(df[['defect_type', 'defect_location', 'severity', 'inspection_method', 'product_id', 'repair_cost']], n_samples=n_synthetic_samples)

    print("Augmenting data...")
    # Ensure columns match before concatenating
    # Align columns, handling potentially missing ones in synthetic data (like defect_date)
    common_cols = list(set(df.columns) & set(synthetic_df.columns))
    augmented_df = pd.concat([df[common_cols], synthetic_df[common_cols]], ignore_index=True)


    print("Handling outliers in 'repair_cost'...")
    q_low = augmented_df["repair_cost"].quantile(0.05)
    q_high = augmented_df["repair_cost"].quantile(0.95)
    df_filtered = augmented_df[(augmented_df["repair_cost"] >= q_low) & (augmented_df["repair_cost"] <= q_high)].copy()


    print("Processing product IDs...")
    df_filtered['product_id'] = df_filtered['product_id'].astype(str) # Ensure string type
    top_products = df_filtered['product_id'].value_counts().nlargest(top_n_products).index
    df_filtered['product_id'] = df_filtered['product_id'].apply(lambda x: x if x in top_products else '기타')

    # Convert defect_date again after concat if it exists
    if 'defect_date' in df_filtered.columns:
        df_filtered['defect_date'] = pd.to_datetime(df_filtered['defect_date'], errors='coerce')


    # Target transformation
    df_filtered['repair_cost_log'] = np.log1p(df_filtered['repair_cost'])

    print(f"Data prepared. Shape: {df_filtered.shape}")
    return df_filtered

# --- Feature Engineering ---
def engineer_features(df):
    """Creates new features from existing ones."""
    print("Engineering features...")
    df_eng = df.copy()

    # Date features (handle missing dates)
    if 'defect_date' in df_eng.columns and pd.api.types.is_datetime64_any_dtype(df_eng['defect_date']):
        df_eng['defect_year'] = df_eng['defect_date'].dt.year
        df_eng['defect_month'] = df_eng['defect_date'].dt.month
        df_eng['defect_day'] = df_eng['defect_date'].dt.day
        df_eng['defect_weekday'] = df_eng['defect_date'].dt.weekday
        df_eng['defect_weekofyear'] = df_eng['defect_date'].dt.isocalendar().week.astype(int)
        # Calculate age relative to a fixed recent date or max date in data for consistency
        latest_date = df_eng['defect_date'].max() if not df_eng['defect_date'].isnull().all() else pd.Timestamp.today()
        df_eng['defect_age_days'] = (latest_date - df_eng['defect_date']).dt.days
        # Fill NaN values created from NaT dates
        date_feature_cols = ['defect_year', 'defect_month', 'defect_day', 'defect_weekday', 'defect_weekofyear', 'defect_age_days']
        for col in date_feature_cols:
            if col in df_eng.columns: # Check if column was created
                # Sensible imputation (e.g., median, mean, or -1 for missing indicator)
                df_eng[col] = df_eng[col].fillna(df_eng[col].median() if pd.api.types.is_numeric_dtype(df_eng[col]) else -1)

    else:
        print("Skipping date feature engineering as 'defect_date' is missing or not datetime.")


    # --- Placeholder for other feature engineering ideas ---
    # Example: Interaction features (uncomment and adapt if needed)
    # df_eng['type_location'] = df_eng['defect_type'] + "_" + df_eng['defect_location']

    # Drop original date column and intermediate columns if necessary
    columns_to_drop = ['defect_date', 'repair_cost'] # Drop original target, keep log target
    columns_to_drop = [col for col in columns_to_drop if col in df_eng.columns] # Drop only if they exist
    df_eng = df_eng.drop(columns=columns_to_drop)


    print(f"Features engineered. Shape: {df_eng.shape}")
    return df_eng

# --- Preprocessing Pipeline ---
def create_preprocessor(df):
    """Creates a ColumnTransformer for preprocessing."""
    print("Defining preprocessor...")

    # Identify column types automatically after feature engineering
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = df.select_dtypes(include=np.number).columns.tolist()
    # Exclude the target variable ('repair_cost_log') from features
    if 'repair_cost_log' in numerical_features:
        numerical_features.remove('repair_cost_log')
    if 'repair_cost_log' in categorical_features: # Should not happen, but check
        categorical_features.remove('repair_cost_log')

    # Define transformers
    # Note: TF-IDF is applied only to 'defect_type'. If 'defect_description' were used,
    # it would likely be added here or handled separately with more advanced NLP.
    # For simplicity, we only include 'defect_type' if it's categorical.
    text_feature = 'defect_type' if 'defect_type' in categorical_features else None
    if text_feature:
        categorical_features.remove(text_feature) # Remove from OHE list
        transformers = [
            ('text', TfidfVectorizer(max_features=100), text_feature),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), categorical_features), # Ensure sparse output
            ('num', StandardScaler(), numerical_features)
        ]
    else: # No text feature identified
        transformers = [
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), categorical_features),
            ('num', StandardScaler(), numerical_features)
        ]

    # Filter out transformers with empty feature lists
    transformers = [(name, trans, cols) for name, trans, cols in transformers if cols]


    # Define ColumnTransformer
    preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough') # Keep other columns if any


    # Helper function to make output dense if needed by certain models (LGBM handles sparse)
    def to_dense(X):
        return X.toarray()
    dense_transformer = FunctionTransformer(to_dense, accept_sparse=True)

    # --- Create the full pipeline ---
    # Note: If using models that require dense input, add 'dense_transformer' step
    # model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
    #                                   ('todense', dense_transformer), # Uncomment if needed
    #                                   ('regressor', lgb.LGBMRegressor(random_state=42))])
    # Since LGBM handles sparse matrices well, we don't need to dense it.
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('regressor', lgb.LGBMRegressor(random_state=42))])


    # Identify feature names after transformation (useful for feature importance)
    # Note: This is complex with ColumnTransformer. We'll get importance from LGBM directly later.


    print("Preprocessor defined.")
    return model_pipeline, numerical_features, categorical_features # Return feature lists too

# --- Model Training and Evaluation ---
def train_evaluate_model(pipeline, X, y, n_splits=5, random_state=42):
    """Trains and evaluates the model using K-Fold cross-validation and RandomizedSearch."""
    print("Starting model training and evaluation with K-Fold Cross-Validation...")
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Define parameter distribution for RandomizedSearch
    param_dist = {
        'regressor__n_estimators': sp_randint(100, 2000),
        'regressor__learning_rate': sp_uniform(0.01, 0.2),
        'regressor__num_leaves': sp_randint(20, 60),
        'regressor__max_depth': [-1, 5, 10, 15, 20], # -1 means no limit
        'regressor__subsample': sp_uniform(0.7, 0.3), # range is [loc, loc + scale]
        'regressor__colsample_bytree': sp_uniform(0.7, 0.3)
    }

    # Define RMSLE scorer
    def rmsle_func(y_true, y_pred):
        # Ensure no negative predictions if model sometimes outputs small negatives
        y_pred_clipped = np.maximum(y_pred, 0)
        return np.sqrt(mean_squared_log_error(y_true, y_pred_clipped))

    # Scorer needs log-transformed target, but we want RMSLE on original scale
    # We'll calculate RMSLE manually after prediction. Use RMSE on log scale for tuning.
    scorers = {
        'neg_root_mean_squared_error': 'neg_root_mean_squared_error', # Lower is better
        'r2': 'r2'
    }


    # RandomizedSearchCV with the pipeline
    print("Performing Randomized Search CV...")
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=30, # Number of parameter settings that are sampled
        scoring=scorers,
        refit='neg_root_mean_squared_error', # Refit using the best RMSE score
        cv=cv,
        random_state=random_state,
        n_jobs=-1, # Use all available cores
        verbose=1 # Show progress
    )

    random_search.fit(X, y) # y is log-transformed target

    print(f"\nBest Parameters found: {random_search.best_params_}")
    print(f"Best CV Score (Neg RMSE on log scale): {random_search.best_score_:.4f}")


    # --- Evaluate the best model found by RandomizedSearch on the full dataset (using CV predictions) ---
    # Or evaluate on a separate hold-out set if available. Here we average CV scores.
    # Note: random_search.cv_results_ contains detailed scores for each fold and param set.
    cv_results_df = pd.DataFrame(random_search.cv_results_)
    best_index = random_search.best_index_


    # Averaged metrics from cross-validation for the best estimator
    avg_rmse_log = -cv_results_df.loc[best_index, 'mean_test_neg_root_mean_squared_error']
    avg_r2_log = cv_results_df.loc[best_index, 'mean_test_r2']


    print("\n--- Averaged Cross-Validation Metrics (on log-transformed scale) ---")
    print(f"Average RMSE (log scale): {avg_rmse_log:.4f}")
    print(f"Average R^2 (log scale):  {avg_r2_log:.4f}")


    # To get metrics on the original scale, we need predictions from each fold's test set
    # This requires manually looping through CV or using cross_val_predict
    from sklearn.model_selection import cross_val_predict
    print("\nCalculating metrics on original scale using cross_val_predict...")
    best_pipeline = random_search.best_estimator_
    y_pred_log_cv = cross_val_predict(best_pipeline, X, y, cv=cv, n_jobs=-1)


    # Inverse transform predictions and true values
    y_pred_actual_cv = np.expm1(y_pred_log_cv)
    y_actual = np.expm1(y)


    # Calculate metrics on the original scale
    mse_actual = mean_squared_error(y_actual, y_pred_actual_cv)
    mae_actual = mean_absolute_error(y_actual, y_pred_actual_cv)
    r2_actual = r2_score(y_actual, y_pred_actual_cv)
    # Calculate RMSLE on the original scale
    # Clip predictions to avoid issues with log(negative) if any occurred, though expm1 should handle it.
    y_pred_actual_cv_clipped = np.maximum(y_pred_actual_cv, 0)
    rmsle_actual = np.sqrt(mean_squared_log_error(y_actual, y_pred_actual_cv_clipped))


    print("\n--- Averaged Cross-Validation Metrics (on original scale) ---")
    print(f"Mean Squared Error (MSE): {mse_actual:.4f}")
    print(f"Mean Absolute Error (MAE): {mae_actual:.4f}")
    print(f"R^2 Score:                {r2_actual:.4f}")
    print(f"Root Mean Squared Log Error (RMSLE): {rmsle_actual:.4f}")


    return random_search.best_estimator_, y_actual, y_pred_actual_cv

# --- Visualization ---
def plot_results(y_true, y_pred):
    """Visualizes actual vs predicted values."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Ideal Prediction (y=x)')
    plt.xlabel('실제 수리 비용 (원래 스케일)')
    plt.ylabel('예측 수리 비용 (원래 스케일 - CV 예측)')
    plt.title('실제 vs 예측 수리 비용 (Cross-Validation 결과)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    DATA_URL = "https://raw.githubusercontent.com/Dong-yeon/Photon-Server-Test/refs/heads/main/defects_data.csv"

    # 1. Load and Prepare Data
    df_prepared = load_and_prepare_data(DATA_URL, n_synthetic_samples=500)

    # 2. Feature Engineering
    df_engineered = engineer_features(df_prepared)

    # Define Features (X) and Target (y)
    target_column = 'repair_cost_log'
    if target_column not in df_engineered.columns:
        raise ValueError(f"Target column '{target_column}' not found after feature engineering.")

    y = df_engineered[target_column]
    X = df_engineered.drop(columns=[target_column])

    # Ensure X doesn't contain the original repair_cost either
    if 'repair_cost' in X.columns:
        X = X.drop(columns=['repair_cost'])


    # 3. Create Preprocessor and Pipeline
    # Pass X (features only) to define the preprocessor based on feature types
    full_pipeline, num_features, cat_features = create_preprocessor(X)


    # 4. Train and Evaluate Model using Cross-Validation and Randomized Search
    best_model_pipeline, y_actual, y_pred_actual_cv = train_evaluate_model(full_pipeline, X, y, n_splits=5)


    # 5. Visualize Results (using CV predictions)
    plot_results(y_actual, y_pred_actual_cv)


    # 6. Feature Importance (from the best model found)
    try:
        # Access the regressor step in the final pipeline
        lgbm_model = best_model_pipeline.named_steps['regressor']

        # Access the preprocessor step to get feature names if possible
        preprocessor = best_model_pipeline.named_steps['preprocessor']

        # Get feature names from the preprocessor
        # This requires careful handling based on the transformers used
        feature_names_out = []
        for name, transformer, columns in preprocessor.transformers_:
            if name == 'remainder' and transformer == 'passthrough':
                # Get names for passthrough columns if any (need slicing logic if used)
                pass # Adapting this requires knowing exactly what columns are passed through
            elif hasattr(transformer, 'get_feature_names_out'):
                if isinstance(columns, str): # Single column like TFIDF
                    cols_list = [columns]
                else: # List of columns
                    cols_list = list(columns)
                # Prefix feature names if needed (e.g., OHE)
                feature_names_out.extend(transformer.get_feature_names_out(cols_list))
            elif name == 'num': # StandardScaler doesn't change names
                feature_names_out.extend(columns)


        if feature_names_out and len(feature_names_out) == lgbm_model.n_features_:
            importance_df = pd.DataFrame({
                'feature': feature_names_out,
                'importance': lgbm_model.feature_importances_
            }).sort_values(by='importance', ascending=False)


            print("\n--- Feature Importance (from best model) ---")
            print(importance_df.head(20)) # Display top 20 features


            # Plot feature importance
            plt.figure(figsize=(10, 8))
            sns.barplot(x='importance', y='feature', data=importance_df.head(20))
            plt.title('Top 20 Feature Importances (LightGBM)')
            plt.tight_layout()
            plt.show()
        else:
            print("\nCould not automatically retrieve all feature names for importance plot.")
            print("Number of names found:", len(feature_names_out) if feature_names_out else 0)
            print("Number of features in model:", lgbm_model.n_features_)
            # Print default importance if names don't match
            print("\nRaw Feature Importances (indices):", lgbm_model.feature_importances_)


    except Exception as e:
        print(f"\nError retrieving feature importance: {e}")

    print("\nScript finished.")
