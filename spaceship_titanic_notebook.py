# %% [markdown]
# # Spaceship Titanic - End-to-End Kaggle Solution
#
# This notebook is written like a classroom walkthrough.
# After each code cell, I explain:
# - what the cell does
# - why we need it
# - why I chose this approach
#
# Goal:
# Predict whether a passenger was `Transported` (`True` or `False`).
#
# Expected files:
# - `train.csv`
# - `test.csv`
# - `sample_submission.csv`
#
# Put them either:
# - in the same folder as this notebook/script, or
# - inside a `data/` folder.

# %% [markdown]
# ## Cell 1: Import the libraries
#
# We start by importing the tools we need for:
# - data handling (`pandas`, `numpy`)
# - plotting (`matplotlib`, `seaborn`)
# - preprocessing and modeling (`scikit-learn`)

# %%
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)
sns.set_theme(style="whitegrid")

# %% [markdown]
# ### Why this cell?
#
# - `pandas` is the main library for reading and manipulating CSV files.
# - `numpy` helps with numerical operations.
# - `matplotlib` and `seaborn` are for visual understanding of the data.
# - `scikit-learn` gives us the machine learning pipeline.
#
# Why this approach:
# I chose `Pipeline` + `ColumnTransformer` because this is one of the cleanest and safest ways to do ML.
# It helps us avoid mistakes like preprocessing train and test data differently.

# %% [markdown]
# ## Cell 2: Load the dataset
#
# This cell automatically looks for the CSV files in common locations so the notebook is easier to run.

# %%
DATA_CANDIDATES = [
    Path("."),
    Path("data"),
    Path("/kaggle/input/i-2526-spaceship-titanic"),
    Path("/kaggle/input/spaceship-titanic"),
]


def find_data_dir(candidates):
    for folder in candidates:
        if (folder / "train.csv").exists() and (folder / "test.csv").exists():
            return folder
    raise FileNotFoundError(
        "Could not find train.csv and test.csv. Place the Kaggle files in this folder or in a data/ folder."
    )


DATA_DIR = find_data_dir(DATA_CANDIDATES)

train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")
sample_submission = pd.read_csv(DATA_DIR / "sample_submission.csv")

print("Data directory:", DATA_DIR.resolve())
print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)

# %% [markdown]
# ### Why this cell?
#
# We need to load the training and test sets before doing anything else.
#
# Why I wrote it this way:
# Many beginners hardcode one path and later the notebook breaks on another machine.
# This version is more flexible:
# - local folder
# - `data/` folder
# - Kaggle notebook input path

# %% [markdown]
# ## Cell 3: Take a quick look at the data
#
# Before modeling, we should understand:
# - what columns exist
# - what the values look like
# - whether there are missing values

# %%
display(train_df.head())
display(train_df.info())
display(train_df.isnull().sum().sort_values(ascending=False))

# %% [markdown]
# ### Why this cell?
#
# This is basic but very important.
# Good ML starts with understanding the data, not with jumping directly into model training.
#
# What we learn here:
# - which columns are categorical
# - which columns are numeric
# - which columns have missing values
#
# Why I chose this:
# `head()`, `info()`, and null counts are the fastest first checks in almost every tabular ML project.

# %% [markdown]
# ## Cell 4: Check the target balance
#
# We want to see whether `Transported` is balanced or imbalanced.

# %%
target_counts = train_df["Transported"].value_counts(normalize=True)
print(target_counts)

plt.figure(figsize=(6, 4))
sns.countplot(data=train_df, x="Transported")
plt.title("Target Distribution")
plt.show()

# %% [markdown]
# ### Why this cell?
#
# If one class is much larger than the other, accuracy can become misleading.
#
# Why I checked this:
# For this competition, the target is usually fairly balanced.
# That means accuracy is a reasonable metric, and we can use standard classification models confidently.

# %% [markdown]
# ## Cell 5: Create better features
#
# Raw columns are useful, but some columns contain hidden structure:
# - `PassengerId` contains a group id
# - `Cabin` contains deck, cabin number, and ship side
# - spending columns can be combined
# - people in CryoSleep often have zero spending
#
# We will create extra features from these patterns.

# %%
EXPENSE_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


def add_features(df):
    df = df.copy()

    passenger_parts = df["PassengerId"].str.split("_", expand=True)
    df["GroupId"] = passenger_parts[0]
    df["GroupNumber"] = pd.to_numeric(passenger_parts[1], errors="coerce")

    group_sizes = df["GroupId"].value_counts()
    df["GroupSize"] = df["GroupId"].map(group_sizes)
    df["IsAlone"] = (df["GroupSize"] == 1).astype(int)

    cabin_parts = df["Cabin"].fillna("Unknown/0/U").str.split("/", expand=True)
    df["CabinDeck"] = cabin_parts[0]
    df["CabinNum"] = pd.to_numeric(cabin_parts[1], errors="coerce")
    df["CabinSide"] = cabin_parts[2]

    df["LastName"] = df["Name"].fillna("Unknown Unknown").str.split().str[-1]
    last_name_counts = df["LastName"].value_counts()
    df["LastNameSize"] = df["LastName"].map(last_name_counts)

    df["TotalSpend"] = df[EXPENSE_COLS].fillna(0).sum(axis=1)
    df["NoSpend"] = (df["TotalSpend"] == 0).astype(int)

    cryo_map = {"True": 1, "False": 0, True: 1, False: 0}
    df["CryoSleepFlag"] = df["CryoSleep"].map(cryo_map)
    df["VipFlag"] = df["VIP"].map(cryo_map)

    df["AgeBucket"] = pd.cut(
        df["Age"],
        bins=[-1, 12, 18, 25, 40, 60, 100],
        labels=["Child", "Teen", "YoungAdult", "Adult", "MiddleAge", "Senior"],
    )

    df["SpendingPerAge"] = df["TotalSpend"] / (df["Age"].fillna(df["Age"].median()) + 1)

    return df


train_fe = add_features(train_df)
test_fe = add_features(test_df)

display(train_fe.head())

# %% [markdown]
# ### Why this cell?
#
# This is one of the most valuable parts of the solution.
#
# Why these features:
# - `GroupId`, `GroupSize`, `IsAlone`:
#   people traveling together often share behavior patterns.
# - `CabinDeck`, `CabinNum`, `CabinSide`:
#   location on the ship may matter.
# - `TotalSpend`, `NoSpend`:
#   strong signal because passengers in CryoSleep usually do not spend money.
# - `AgeBucket`:
#   sometimes models learn better from grouped age behavior than raw age alone.
# - `LastNameSize`:
#   can weakly capture family/group behavior.
#
# Why I chose feature engineering here:
# In tabular Kaggle competitions, clever feature engineering often improves results more than blindly changing models.

# %% [markdown]
# ## Cell 6: Separate features and target
#
# We split training data into:
# - `X` = input columns
# - `y` = target column

# %%
X = train_fe.drop(columns=["Transported"])
y = train_fe["Transported"].astype(int)
X_test = test_fe.copy()

print("Training features shape:", X.shape)
print("Target shape:", y.shape)

# %% [markdown]
# ### Why this cell?
#
# Machine learning models need the input features separated from the output target.
#
# Why convert target to integers:
# Some models work more consistently when the target is represented as `0` and `1`.
# Here:
# - `True` becomes `1`
# - `False` becomes `0`

# %% [markdown]
# ## Cell 7: Decide which columns are numeric and categorical
#
# Different column types need different preprocessing.

# %%
numeric_features = [
    "Age",
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
    "GroupNumber",
    "GroupSize",
    "CabinNum",
    "LastNameSize",
    "TotalSpend",
    "NoSpend",
    "CryoSleepFlag",
    "VipFlag",
    "IsAlone",
    "SpendingPerAge",
]

categorical_features = [
    "HomePlanet",
    "CryoSleep",
    "Cabin",
    "Destination",
    "VIP",
    "GroupId",
    "CabinDeck",
    "CabinSide",
    "LastName",
    "AgeBucket",
]

print("Numeric columns:", numeric_features)
print("Categorical columns:", categorical_features)

# %% [markdown]
# ### Why this cell?
#
# We cannot treat all columns the same way.
#
# Why this split matters:
# - numeric columns need imputation and sometimes scaling
# - categorical columns need imputation and encoding
#
# Why I still kept some original columns like `Cabin` and `CryoSleep`:
# Sometimes the model gains extra signal from both:
# - the original raw column
# - the engineered sub-parts

# %% [markdown]
# ## Cell 8: Build preprocessing pipelines
#
# We define:
# - how to fill missing numeric values
# - how to fill missing categorical values
# - how to convert text categories into machine-readable form

# %%
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

preprocessor

# %% [markdown]
# ### Why this cell?
#
# This cell defines how raw data becomes model-ready data.
#
# Why these choices:
# - numeric missing values: `median` is robust and simple
# - categorical missing values: `most_frequent` is a strong baseline
# - `OneHotEncoder`: standard choice for non-ordered categories
# - `handle_unknown="ignore"`: very important so the pipeline does not break on unseen test categories
#
# Why scaling:
# Scaling helps linear models like Logistic Regression.
# It is not strictly required for tree models, but keeping one shared preprocessing pipeline makes comparison easier.

# %% [markdown]
# ## Cell 9: Create multiple candidate models
#
# Instead of choosing one model blindly, we compare a few strong baselines.

# %%
models = {
    "LogisticRegression": LogisticRegression(max_iter=2000, random_state=42),
    "RandomForest": RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ),
    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for model_name, model in models.items():
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    )

    results.append(
        {
            "Model": model_name,
            "CV Mean Accuracy": scores.mean(),
            "CV Std": scores.std(),
        }
    )

results_df = pd.DataFrame(results).sort_values("CV Mean Accuracy", ascending=False)
display(results_df)

# %% [markdown]
# ### Why this cell?
#
# This is a very important habit in machine learning:
# do not assume the best model before testing.
#
# Why these models:
# - `LogisticRegression`: simple, fast, strong baseline
# - `RandomForest`: handles nonlinear relationships well
# - `ExtraTrees`: often performs very well on tabular data
#
# Why cross-validation:
# One train/validation split can be lucky or unlucky.
# Cross-validation gives a more reliable estimate by testing on multiple folds.
#
# Why accuracy:
# Kaggle Spaceship Titanic is commonly evaluated with accuracy, and the target is fairly balanced.

# %% [markdown]
# ## Cell 10: Select the best model and train on all training data
#
# We now pick the model with the highest cross-validation score and fit it on the full training set.

# %%
best_model_name = results_df.iloc[0]["Model"]
best_model = clone(models[best_model_name])

final_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", best_model),
    ]
)

final_pipeline.fit(X, y)
print("Best model selected:", best_model_name)

# %% [markdown]
# ### Why this cell?
#
# After comparing models fairly, we train the winner on all available training data.
#
# Why fit on the full data now:
# During cross-validation, each model only sees part of the data at a time.
# Once we choose the best model, we want it to learn from the entire training set before predicting on the test set.

# %% [markdown]
# ## Cell 11: Optional training-set sanity check
#
# This is not the Kaggle score.
# It is just a quick check to see whether the model has learned meaningful patterns.

# %%
train_predictions = final_pipeline.predict(X)
train_accuracy = accuracy_score(y, train_predictions)
print("Training accuracy:", round(train_accuracy, 4))

# %% [markdown]
# ### Why this cell?
#
# This gives us a quick sanity check.
#
# Important note:
# Training accuracy is usually optimistic.
# We should trust cross-validation more than this number.
#
# Why still include it:
# It helps beginners see whether the model is learning at all.
# If training accuracy were very low, that would be a warning sign.

# %% [markdown]
# ## Cell 12: Predict on the Kaggle test set
#
# Now we generate predictions for the unseen passengers.

# %%
test_predictions = final_pipeline.predict(X_test)
test_predictions = pd.Series(test_predictions).astype(bool)

submission = pd.DataFrame(
    {
        "PassengerId": test_df["PassengerId"],
        "Transported": test_predictions,
    }
)

display(submission.head())

# %% [markdown]
# ### Why this cell?
#
# Kaggle expects predictions for every row in `test.csv`.
#
# Why convert back to boolean:
# The competition submission format uses `True` and `False`, not `1` and `0`.
# So we convert the model output into the required format.

# %% [markdown]
# ## Cell 13: Save the submission file
#
# This creates the CSV that you can upload to Kaggle.

# %%
output_path = Path("submission.csv")
submission.to_csv(output_path, index=False)
print(f"Submission file saved to: {output_path.resolve()}")

# %% [markdown]
# ### Why this cell?
#
# This is the final step of a Kaggle workflow.
#
# Why the filename is `submission.csv`:
# It is the standard convention, easy to recognize, and ready to upload directly.

# %% [markdown]
# ## Cell 14: What to try next for a better score
#
# These are common improvement ideas once the baseline works well.

# %%
improvement_ideas = [
    "Tune RandomForest and ExtraTrees hyperparameters more carefully.",
    "Try CatBoost or XGBoost if allowed in your environment.",
    "Create stronger group-level features using PassengerId and surname patterns.",
    "Impute CryoSleep and VIP using business logic before modeling.",
    "Use ensembling: average or vote between top-performing models.",
]

for i, idea in enumerate(improvement_ideas, start=1):
    print(f"{i}. {idea}")

# %% [markdown]
# ### Why this cell?
#
# A good Kaggle notebook should not stop at one working solution.
# It should also show the path to improvement.
#
# Why these next steps:
# - tuning improves model behavior
# - better feature engineering often boosts tabular performance
# - ensembling is a classic Kaggle strategy
#
# For learning purposes, I started with a solution that is:
# - clean
# - explainable
# - strong enough to be competitive as a solid baseline
# - easy for a student to extend
