
# This program implements A* for solving a sliding tile puzzle

import numpy as np
import queue

class PuzzleState():
    SOLVED_PUZZLE = np.arange(9).reshape((3, 3))

    def __init__(self,conf,g,predState):
        self.puzzle = conf     # Configuration of the state
        self.gcost = g         # Path cost
        self._compute_heuristic_cost()  # Set heuristic cost
        self.fcost = self.gcost + self.hcost
        self.pred = predState  # Predecesor state
        self.zeroloc = np.argwhere(self.puzzle == 0)[0]
        self.action_from_pred = None
    
    def __hash__(self):
        return tuple(self.puzzle.ravel()).__hash__()
    
    def _compute_heuristic_cost(self):
        self.hcost = sum(np.argwhere(self.puzzle ==0)[0])
        #Because 0,0 is the goal, can just sum the array of the coordinates
        return self.hcost
        #formula normally should be (abs(start - goal) + abs(start - goal))
    
    def is_goal(self):
        return np.array_equal(PuzzleState.SOLVED_PUZZLE,self.puzzle)
    
    def __eq__(self, other):
        return np.array_equal(self.puzzle, other.puzzle)
    
    def __lt__(self, other):
        return self.fcost < other.fcost
    
    def __str__(self):
        return np.str(self.puzzle)
    
    move = 0
    
    def show_path(self):
        if self.pred is not None:
            self.pred.show_path()
        
        if PuzzleState.move==0:
            print('START')
        else:
            print('Move',PuzzleState.move, 'ACTION:', self.action_from_pred)
        PuzzleState.move = PuzzleState.move + 1
        print(self)
    
    def can_move(self, direction):
        if self.zeroloc[1] == 0 and direction == "left":
            return False
        if self.zeroloc[1] == 2 and direction == "right":
            return False
        if self.zeroloc[0] == 0 and direction == "up":
            return False
        if self.zeroloc[0] == 2 and direction == "down":
            return False
        return True

    def gen_next_state(self, direction):
        y = self.zeroloc[0]
        x = self.zeroloc[1]
        if direction == "left":
            self.puzzle[y][x] = self.puzzle[y][x-1]
            self.puzzle[y][x-1] = 0
        if direction == "up":
            self.puzzle[y][x] = self.puzzle[y-1][x]
            self.puzzle[y-1][x] = 0
        if direction == "down":
            self.puzzle[y][x] = self.puzzle [y+1][x]
            self.puzzle[y+1][x] = 0
        if direction == "right":
            self.puzzle[y][x] = self.puzzle[y][x+1]
            self.puzzle[y][x+1] = 0
        return self.puzzle


            

print('Artificial Intelligence')
print('MP1: A* for Sliding Puzzle')
print('SEMESTER: Fall 2019 - Block 1')
print('NAME: Scott Carrington')
print()

# load random start state onto frontier priority queue
frontier = queue.PriorityQueue()
  
  
a = np.loadtxt('mp1input.txt', dtype=np.int32)
start_state = PuzzleState(a,0,None)



frontier.put(start_state)

closed_set = set()

num_states = 0
while not frontier.empty():
    #  choose state at front of priority queue
    next_state = frontier.get()
    
    #  if goal then quit and return path
    if next_state.is_goal():
        next_state.show_path()
        break
    
    # Add state chosen for expansion to closed_set
    closed_set.add(next_state)
    num_states = num_states + 1
    
    # Expand state (up to 4 moves possible)
    possible_moves = ['up','down','left','right']
    for move in possible_moves:
        if next_state.can_move(move):
            neighbor = next_state.gen_next_state(move)
            if neighbor in closed_set:
                continue
            if neighbor not in frontier.queue:                           
                frontier.put(neighbor)
            # If it's already in the frontier, it's gauranteed to have lower cost, so no need to update

print('\nNumber of states visited =',num_states)
