"""
Writes requirements.txt using the versions installed on YOUR computer.
Run it once from the project folder (with your virtual environment active):

    python make_requirements.py

Why: the saved model was trained with these exact versions, and the deployed app
must use the same ones or the model may refuse to load.
"""
from importlib.metadata import PackageNotFoundError, version

PACKAGES = ["streamlit", "pandas", "numpy", "scikit-learn", "xgboost", "shap", "matplotlib", "joblib"]

lines = []
for name in PACKAGES:
    try:
        lines.append(f"{name}=={version(name)}")
    except PackageNotFoundError:
        print(f"WARNING: {name} is not installed here, so it was written without a version.")
        lines.append(name)

with open("requirements.txt", "w") as f:
    f.write("\n".join(lines) + "\n")

print("Wrote requirements.txt:")
print("\n".join(lines))
