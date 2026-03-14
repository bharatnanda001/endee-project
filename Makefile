.PHONY: install run-api run-ui mock-db test build up down pre-commit align

install:
	pip install -r requirements.txt

run-api:
	uvicorn app:app --reload --port 8000

run-ui:
	streamlit run ui.py

mock-db:
	python mock_endee.py

test:
	python test_pipeline.py

up:
	docker-compose up -d --build

down:
	docker-compose down

ingest:
	python ingest.py

format:
	pip install black isort
	black .
	isort .
