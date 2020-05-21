# differential-evolution

## Installation

1. Install required packages
   ```
   pip install -r requirements.txt
   ```
2. Compile testing code in `cec17_test_func.c`
   ```
   gcc -fPIC -shared -lm -o diff_evolution/cec17_test_func.so diff_evolution/cec17_test_func.c
   ```
3. Run tests
   ```
   python -m pytest
   ```
