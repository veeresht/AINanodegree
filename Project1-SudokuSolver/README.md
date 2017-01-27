# Artificial Intelligence Nanodegree
## Introductory Project: Diagonal Sudoku Solver

# Question 1 (Naked Twins)
Q: How do we use constraint propagation to solve the naked twins problem?  
A: Constraint propagation is used in the naked twins problem as follows:  
- First the units containing naked twins (2 boxes having identical sets of only 2 possible candidates) are identified.
- The constraint imposed by a naked twin pair is that none of the other boxes in its unit can have either of the 2 values as candidates.
- Hence such values can be eliminated from the peers of the naked twin pair in such units.
- This can have the effect that some boxes are left with only a single choice after such elimination and hence are solved or also other naked twin pairs are created.
- Iterating the eliminate(), naked_twins(), only_choice() strategies forms our constraint propagation
part of sudoku solving algorithm.  

# Question 2 (Diagonal Sudoku)
Q: How do we use constraint propagation to solve the diagonal sudoku problem?  
A: Constraint propagation is used to solve the diagonal sudoku problem as follows:
- Identify boxes belonging to the two diagonal units.
- In eliminate(), naked_twins() and only_choice() strategies we consider the two diagonal units along with row, column, square units as part of constraint propagation for solving the diagonal sudoku problem.

### Install

This project requires **Python 3**.

We recommend students install [Anaconda](https://www.continuum.io/downloads), a pre-packaged Python distribution that contains all of the necessary libraries and software for this project.
Please try using the environment we provided in the Anaconda lesson of the Nanodegree.

##### Optional: Pygame

Optionally, you can also install pygame if you want to see your visualization. If you've followed our instructions for setting up our conda environment, you should be all set.

If not, please see how to download pygame [here](http://www.pygame.org/download.shtml).

### Code

* `solutions.py` - You'll fill this in as part of your solution.
* `solution_test.py` - Do not modify this. You can test your solution by running `python solution_test.py`.
* `PySudoku.py` - Do not modify this. This is code for visualizing your solution.
* `visualize.py` - Do not modify this. This is code for visualizing your solution.

### Visualizing

To visualize your solution, please only assign values to the values_dict using the ```assign_values``` function provided in solution.py

### Data

The data consists of a text file of diagonal sudokus for you to solve.
