# Computation graphs

A package that provides a computational graph and operations on it
## Getting Started

These instructions will give you a copy of the project up and running on
your local machine for development and testing purposes.

### Installing
Download and install package:

     pip install -e compgraph --force-reinstall

You can run one of the examples for illustration. For example, a script that uses a computational graph and counts the number of each word in the text.
   
      python  run_word_count.py "input_file.txt" "output.txt"



## Running the tests

You can run the tests:

    pytest compgraph/tests

### Sample Tests

Tests in the “correctness” folder check the correctness of the computational graph and operations on it.

Tests in the “memory” folder check speed of the computational graph and the memory it occupies.

test_examples checks the correctness of computational graph examples

test_graph, test_operations.py additional tests for computational graph

## Authors

  - **Daniil Domnin** - *Provided all* -
