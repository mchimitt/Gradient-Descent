How to build and run the code:
1) Create a virtual environment
- in the command line, run "pip install virtualenv"
- run "python -m venv venv" to create an environment called env
- run ".\venv\Scripts\activate" to enter into the virtual environment

2) install the necessary packages
- pandas (includes numpy) - pip install pandas
- matplotlib - pip install matplotlib
- scikit learn - pip install -U scikit-learn

3) Running the gradientdescent.py
- Inside the virtual environment, run "python gradientdescent.py"
- When running, matplotlib will pop up with several graphs
    - IMPORTANT TO NOTE: in order to continue with the program, you must close the graph.
    - matplotlib waits for the graph to close before continuing onward.
- After running, you will see information such as the weights, equation, error values, r^2 values, and graphs.
- Note: it may take some time to calculate the weights