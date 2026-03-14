# Contributing to RAG Research Assistant

First off, thank you for considering contributing to the RAG Research Assistant! It's people like you that make the open-source community such an amazing place to learn, inspire, and create.

## Developing Locally
1. Fork and clone the repository.
2. Ensure you have Python 3.10 installed.
3. Install dependencies using: `make install` (or `pip install -r requirements.txt`).
4. Set up the environment: `cp .env.example .env`.
5. Run the local Endee mock server: `make mock-db`.
6. Run the API: `make run-api`.
7. Run the Streamlit UI: `make run-ui`.

## Pull Requests
- Create a new branch for your feature (`git checkout -b feature/amazing-feature`).
- Make your changes and write tests if applicable.
- Make sure all tests pass: `make test`.
- Add meaningful commit messages (e.g., `feat: added new LLM integration`).
- Push your branch and open a Pull Request.

## Issues
Feel free to submit issues and enhancement requests.
