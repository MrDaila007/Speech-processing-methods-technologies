.PHONY: all build-cpp build-python run-cpp run-python run-r test-cpp test-python test-r test clean

all: build-cpp build-python

# === C++ ===

build-cpp:
	g++ -O2 -o cpp/solution cpp/solution.cpp
	g++ -O2 -o cpp/tests cpp/tests.cpp

run-cpp: build-cpp
	cd cpp && ./solution

test-cpp: build-cpp
	cd cpp && ./tests

# === Python ===

build-python:
	python3 -m venv python/venv
	python/venv/bin/pip install --upgrade pip -q

run-python:
	cd python && python3 solution.py

test-python:
	cd python && python3 -m unittest test_solution -v

# === R ===

run-r:
	cd r && Rscript solution.R

test-r:
	cd r && Rscript tests.R

# === All ===

test: test-cpp test-python test-r

clean:
	rm -f cpp/solution cpp/tests
	rm -rf python/venv python/__pycache__
