# RAG PDF Q&A System — Version 1

This repository contains a local-first RAG application prototype for uploading PDF documents, indexing them, and answering questions from the indexed content.

## Goals

- Build the core RAG pipeline locally
- Keep the architecture modular for future versions
- Run the project with Docker Compose

## Project structure

- backend/ — FastAPI application
- frontend/ — React UI
- notebooks/ — trial notebook for pipeline validation
- docker-compose.yml — local container orchestration

## Environment variables

Copy [.env.example](.env.example) to .env and fill in the required values.

## Getting started

1. Create and activate a Python virtual environment
2. Install backend dependencies
3. Set environment variables in .env
4. Run the backend and frontend locally or with Docker Compose

## Notes

This is a learning project and will evolve through multiple versions.
