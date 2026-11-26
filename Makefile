# One command per pipeline step. `make all` reproduces every number and figure.
PY ?= python
export PYTHONPATH := src

.PHONY: all data train evaluate curve monitor cluster figures demo api test docker clean

all: data train evaluate curve monitor cluster figures

data:      ## clean the raw CSV and write the train/val/test split
	$(PY) -m homicide.data

train:     ## baseline + tuned classifiers, select the best on validation
	$(PY) -m homicide.train

evaluate:  ## test-set metrics, bias audit, feature importance, error examples
	$(PY) -m homicide.evaluate

curve:     ## learning curve of the selected model
	$(PY) -m homicide.learning_curve

monitor:   ## drift (KS, chi-square, PSI) and performance per year
	$(PY) -m homicide.monitor

cluster:   ## k-modes cluster detector + Gary, Indiana check
	$(PY) -m homicide.cluster

figures:   ## recreate every figure in reports/figures
	$(PY) -m homicide.figures

demo:      ## batch demo: CSV in, predictions out
	$(PY) -m homicide.predict examples/sample_input.csv examples/sample_output.csv

api:       ## start the REST API on http://localhost:8000/docs
	uvicorn app.main:app --port 8000

test:
	$(PY) -m pytest -q

docker:    ## build and run the API container (needs models/best_model.joblib)
	docker build -t homicide-solvability:1.0 .
	docker run --rm -p 8000:8000 homicide-solvability:1.0

clean:
	rm -rf data/processed/* models/*.joblib reports/metrics/* reports/figures/*
