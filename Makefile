.PHONY: help mlflow baseline

help:
	@echo "make mlflow    - start MLflow server on port 5000"
	@echo "make baseline  - run the baseline experiment"

mlflow:
	mlflow server --port 5000

baseline:
	python src/train.py
