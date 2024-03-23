# DataBhai-ML

## Description

This package is used to load models and provide utilities for DataBhai Application

## Prerequisites

- Python 3.10 +
- GCC Compiler
- CMake 3.29.0 +
- CUDA 11.7 + (Optional)
- Docker (Optional)

## Build Instructions

Follow the build instructions to install this sample project.

1. First, clone the repository from github.
2. Then run the following commands to install necessary libraries and dependencies for the project. Run these commands from project directory:
    ```
    pip install -r requirements.txt
    ```
3. Now to setup the package, run the following commands from project directory:
    ```
    pip install -e .
    ```
4. For hosting your own vector database in local, use docker compose file to setup your own vector database. To start the vector database run the following commands from project directory
   ```
   docker compose up -d
   ```
5. For running pylint tests on project (for linting purposes), run the following commands from project directory
   ```
   pip install -U pylint
   pylint $(git ls-files '*.py')
   ```