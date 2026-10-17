# EvidenceGuard - every result comes from one config and one command.
# The pipeline runs all stages (features -> models -> anomaly -> stability -> ranking) in a single deterministic
# run, so train / stability / rank are aliases of the same command; `ablate` uses the full 3-fold x 5-seed protocol.
PY ?= python
CONFIG ?= configs/default.yaml
DB ?= sqlite:///data/evidenceguard.db

.PHONY: data train stability rank ablate demo api web build-web test test-web copy-db docker

data:            ## write the synthetic dataset + report (IEEE-CIS / PaySim: put the Kaggle files in data/raw/)
	$(PY) -m ml.make_data --config $(CONFIG)

train stability rank:   ## one config in -> results/<run_id>/ + database rows out
	$(PY) -m ml.run_experiment --config $(CONFIG) --db $(DB)

ablate:          ## full protocol: 3 chronological folds x 5 seeds, 2000 bootstrap resamples
	$(PY) -m ml.run_experiment --config configs/full.yaml --db $(DB)

demo:            ## rebuild the small shipped demo database (data/demo.db)
	$(PY) -m ml.run_experiment --config configs/demo.yaml --run-id demo --db sqlite:///data/demo.db

api:             ## API on :8000 (also serves web/dist when built)
	DATABASE_URL=$${DATABASE_URL:-sqlite:///data/demo.db} $(PY) -m uvicorn api.main:app --reload --port 8000

web:             ## dashboard dev server on :5173 (proxies /api to :8000)
	cd web && npm install && npm run dev

build-web:
	cd web && npm install && npm run build

test:
	$(PY) -m pytest -q

test-web:
	cd web && npm install && npm test

copy-db:         ## copy the demo database into DATABASE_URL (e.g. PostgreSQL)
	$(PY) -m ml.copy_db --src sqlite:///data/demo.db --dst $(DATABASE_URL) --if-empty

docker:          ## PostgreSQL + API + web
	docker compose up --build
