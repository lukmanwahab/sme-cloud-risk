.PHONY: install test run quick score clean

install:        ## editable install with dev extras
	pip install -e ".[dev]"

test:           ## run the test suite
	pytest -q

run:            ## full pipeline with nested tuning and 1000 bootstraps (~2-3 min)
	cloudrisk run

quick:          ## fast pipeline without tuning (~20 s)
	cloudrisk run --no-tune --n-boot 200

score:          ## score the example new-firm file with the saved model
	cloudrisk score --input data/new_firms_example.csv --output reports/scored_firms.csv

clean:
	rm -rf models reports/*.csv reports/*.json reports/figures/*.png data/sme_survey_raw.csv
