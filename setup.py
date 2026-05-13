from setuptools import setup, find_packages

setup(
    name="veda",
    version="2.0.0",
    description="VEDA - Autonomous Data Science System",
    author="Keshav Kumar",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "scikit-learn>=1.2.0",
        "xgboost>=1.7.0",
        "lightgbm>=3.3.0",
        "groq>=0.4.0",
        "python-dotenv>=0.21.0",
        "mlflow>=2.9.0",
        "fastapi>=0.109.0",
        "uvicorn>=0.27.0",
        "pydantic>=2.0.0",
    ],
)