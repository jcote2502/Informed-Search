############################################################
# CMPSC 442: Informed Search
############################################################

student_name = "Justin Cote"

############################################################
# Imports
############################################################

# Include your imports here, if any are used.
import random
import copy
from queue import PriorityQueue
import math
from collections import deque


############################################################
# Section 1: Tile Puzzle
############################################################

def create_tile_puzzle(rows, cols):
    board = generate_starting_board(rows,cols)
    return TilePuzzle(board)

def generate_starting_board(rows, cols):
    board = [[i+j * cols + 1 for i in range(cols)] for j in range(rows)]
    board[-1][-1] = 0
    return board

class TilePuzzle(object):
    
    def __init__(self, board):
        self.board = board
        self.rows = len(board)
        self.columns = len(board[0])
        self.empty_cell = self.linear_search(board, 0)
    
    def linear_search(self, bd , target):
        for i in range(len(bd)):
            for j in range(len(bd[i])):
                if bd[i][j] == target:
                    return (i,j)
        return (-1,-1)

    def get_board(self):
        return self.board

    def perform_move(self, direction):
        move_set = {
            "up":(-1,0),
            "down":(1,0),
            "left":(0,-1),
            "right":(0,1)
        }
        empty_row, empty_col = self.empty_cell
        if direction in move_set:
            new_row, new_col = empty_row + move_set[direction][0], empty_col + move_set[direction][1]
            if 0<= new_row < self.rows and 0<= new_col < self.columns:
                self.board[empty_row][empty_col], self.board[new_row][new_col] = self.board[new_row][new_col], self.board[empty_row][empty_col]
                self.empty_cell = (new_row,new_col)
                return True
        return False

    def scramble(self, num_moves):
        move_set = ["up", "left", "down", "right"]
        for _ in range(num_moves):
            ensure_move = False
            while ensure_move == False:
                random_direction = random.choice(move_set)
                ensure_move = self.perform_move(random_direction)

    def is_solved(self):
        goal_position = generate_starting_board(self.rows, self.columns)
        if self.board == goal_position:
            return True
        return False

    def copy(self):
        new_board = copy.deepcopy(self.board)
        new_puzzle = TilePuzzle(new_board)
        return new_puzzle

    def successors(self):
        move_set = ["up", "down", "left", "right"]
        for move in move_set:
            temp_board = self.copy()
            isMove = temp_board.perform_move(move)
            if isMove:
                yield (move, temp_board)

    def find_solutions_iddfs(self):
        solved = False
        limit = 0
        while not solved:
            for move in self.iddfs_helper(limit, []):
                yield move
                solved=True
            limit += 1

    def iddfs_helper(self, limit, moves):
        if self.board == generate_starting_board(self.rows, self.columns):
            yield moves
        elif len(moves) < limit:
            for move, puzzle in self.successors():
                for sol in puzzle.iddfs_helper(limit, moves + [move]):
                    yield sol
    
    def find_solution_a_star(self):
        frontier = PriorityQueue()
        frontier.put((self.manhattan(), 0, [], self))
        visited = set()
        while not frontier.empty():
            node = frontier.get()
            if tuple(tuple(row) for row in node[3].board) in visited:
                continue
            else:
                visited.add(tuple(tuple(row)for row in node[3].board))
            if node[3].is_solved():
                return node[2]
            for move, puzzle in node[3].successors():
                if tuple(tuple(row) for row in puzzle.board)not in visited:
                    frontier.put((node[1]+1+puzzle.manhattan(),node[1]+1, node[2] + [move], puzzle))
        return []

    def manhattan(self):
        cost = 0
        for r in range(self.rows):
            for c in range(self.columns):
                if self.board[r][c] != 0:
                    target_row = (self.board[r][c] - 1) / self.columns
                    target_col = (self.board[r][c] - 1) / self.rows
                    cost += abs(r - target_row) + abs(c - target_col)
        return cost


############################################################
# Section 2: Grid Navigation
############################################################

def find_path(start, goal, scene):
    if scene[start[0]][start[1]] == True or scene[goal[0]][goal[1]] == True:
        return None
    g = GridGame(start,goal,scene)
    return g.solve_grid()

class GridGame():
    def __init__(self,start,goal,scene):
        self.scene = scene
        self.start = start
        self.goal = goal
        self.rows = len(scene)
        self.cols = len(scene[0])
    
    def successors(self):
        move_set = [(-1,0),(1,0),(0,-1),(0,1),(-1,1),(-1,-1),(1,-1),(1,1)]
        for move in move_set:
            new_row, new_col = self.start[0]+move[0], self.start[1]+move[1]
            if 0<= new_row < self.rows and 0<= new_col < self.cols and self.scene[new_row][new_col] == False:
                next_grid = GridGame((new_row,new_col),self.goal,self.scene)
                yield(move,next_grid)
        return move_set

    def euclidean(self):        
        x_dif = (self.start[0] - self.goal[0]) ** 2
        y_dif = (self.start[1] - self.goal[1]) ** 2
        cost = math.sqrt(x_dif+y_dif)
        return cost
    
    def is_solved(self):
        if self.start == self.goal:
            return True
        return False

    def solve_grid(self):
        frontier = PriorityQueue()
        visited = set()
        frontier.put((self.euclidean, 0, [self.start], self))
        while not frontier.empty():
            node = frontier.get()
            if node[3].start in visited:
                continue
            else:
                visited.add(node[3].start)
            if node[3].is_solved():
                return node[2]
            for move, grid in node[3].successors():
                if (grid.start) not in visited:
                    if move in [(-1,0),(1,0),(0,-1),(0,1)]:
                        frontier.put((node[1]+1+grid.euclidean(),node[1]+1, node[2] + [grid.start], grid))
                    else:
                        frontier.put((node[1]+ math.sqrt(2) + grid.euclidean(), node[1] + math.sqrt(2), node[2] + [grid.start], grid ))
        return None


############################################################
# Section 3: Linear Disk Movement, Revisited
############################################################

# this class was copied directly from hw 1
# made few adjustments based off what I've learned is more optimal and effective
class DistinctDiskGrid():
    def __init__ (self,length,n,grid=[]):
        self.length = length
        self.n = n
        self.grid = grid

    def buildGrid(self):
        self.grid = [{i: "disk"} if i < self.n else 0 for i in range(self.length)]
    
    def is_solved(self):
        solution = [0] * (self.length - self.n) + [{i:'disk'} for i in range(self.n-1,-1,-1)]
        if self.grid == solution:
            return True
        return False
    
    def perform_move(self, position, target):
        self.grid[position], self.grid[target] = self.grid[target], self.grid[position]

    def copy(self):
        new_grid = copy.deepcopy(self.grid)
        linear_grid = DistinctDiskGrid(self.length, self.n, new_grid)
        return linear_grid

    def successors(self):
        for i in range(self.length):
            if self.grid[i] != 0:
                # Move to the right 
                if i < self.length - 1 and self.grid[i+1] == 0:
                    new_grid = self.copy()
                    new_grid.perform_move(i,i+1)
                    yield ((i, i+1),'m_right', new_grid)
                # Move to the left
                if i > 0 and self.grid[i - 1] == 0:
                    new_grid = self.copy()
                    new_grid.perform_move(i, i-1)
                    yield ((i, i-1),'m_left', new_grid)
                # Jump to the right
                if i < self.length - 2 and self.grid[i + 2] == 0 and self.grid[i + 1] != 0:
                    new_grid = self.copy()
                    new_grid.perform_move(i, i + 2)
                    yield ((i, i + 2),'j_right', new_grid )
                # Jump to the left
                if i > 1 and self.grid[i - 2] == 0 and self.grid[i - 1] != 0:
                    new_grid = self.copy()
                    new_grid.perform_move(i, i - 2)
                    yield ((i, i - 2),'j_left', new_grid)

    def distance(self):
        total_cost = 0
        # maybe i don't need a target grid ?
        for i in range(self.length):
            if isinstance(self.grid[i], dict):
                dic = self.grid[i]
                key = list(dic.keys())[0]
                total_cost += abs(self.length - 1 - key - i)
        return total_cost

    def solve_disks(self):
        frontier = PriorityQueue()
        visited = []
        frontier.put((self.distance(), 0 , [] , self))
        while not frontier.empty():
            node = frontier.get()
            if node[3].grid in visited:
                continue
            else:
                visited.append([node[3].grid])
            if node[3].is_solved():
                return node[2]
            for move, direction, next_grid in node[3].successors():
                if next_grid not in visited:
                    frontier.put((node[1]+1+next_grid.distance(), node[1]+1, node[2] + [move], next_grid))
        return None


def solve_distinct_disks(length, n):
    grid = DistinctDiskGrid(length,n)
    grid.buildGrid()
    solution = grid.solve_disks()
    return solution



############################################################
# Section 4: Dominoes Game
############################################################

def create_dominoes_game(rows, cols):
    board = []
    for _ in range(rows):
        row = []
        for _ in range(cols):
            row.append(False)
        board.append(row)
    game = DominoesGame(board)
    return game
    pass

class DominoesGame(object):

    def __init__(self, board):
        self.board = board
        self.rows = len(board)
        self.cols = len(board[0])

    def get_board(self):
        return self.board

    def reset(self):
        for i in range(self.rows):
            for j in range(self.cols):
                self.board[i][j] = False

    def is_legal_move(self, row, col, vertical):
        if vertical:
            if row == self.rows - 1 or self.board[row+1][col] or self.board[row][col]:
                return False
        else:
            if col == self.cols - 1 or self.board[row][col+1] or self.board[row][col]:
                return False
        return True

    def legal_moves(self, vertical):
        for i in range(self.rows):
            for j in range(self.cols):
                if self.is_legal_move(i,j,vertical):
                    yield(i,j)

    def perform_move(self, row, col, vertical):
        if self.is_legal_move(row,col,vertical):
            if vertical:
                self.board[row+1][col], self.board[row][col] = True, True
            else:
                self.board[row][col+1], self.board[row][col] = True, True

    def game_over(self, vertical):
        if list(self.legal_moves(vertical)):
            return False
        return True

    def copy(self):
        new_board = copy.deepcopy(self.board)
        new_game = DominoesGame(new_board)
        return new_game

    def successors(self, vertical):
        for row, col in list(self.legal_moves(vertical)):
            new_game = self.copy()
            new_game.perform_move(row,col,vertical)
            yield ((row, col), new_game)

    def get_random_move(self, vertical):
        move_set = list(self.legal_moves(vertical))
        choice = random.choice(move_set)
        return choice
    
    def get_best_move(self, vertical, limit):
        return self.maximum(-float('inf'), float('inf'),limit, vertical, None)
    
    def maximum(self, alpha, beta, limit, vertical, move):
        opponent_moves = list(self.successors(not vertical))
        player_moves = list(self.successors(vertical))
        if self.game_over(vertical) or limit==0:
            return (move, len(player_moves) - len(opponent_moves), 1)
        v_value = -float('inf')
        s_branches = 0
        this_move = move
        for node, game in player_moves:
            _, res_v , res_branches = game.minimum(alpha,beta,limit-1, not vertical, node)
            s_branches += res_branches
            if res_v > v_value:
                v_value = res_v
                this_move = node
            if v_value >= beta:
                return (this_move, v_value, s_branches)
            alpha = max(v_value,alpha)
        
        return (this_move, v_value, s_branches)

    def minimum(self, alpha, beta, limit, vertical, move):
        opponent_moves = list(self.successors(not vertical))
        player_moves = list(self.successors(vertical))
        if self.game_over(vertical) or limit==0:
            return (move, len(opponent_moves) - len(player_moves), 1)
        v_value = float('inf')
        s_branches = 0
        this_move = move
        # player_moves in this function is opponent moves in max function because vertical is negated when passed !!
        for node, game in player_moves:
            _, res_v, res_branches = game.maximum(alpha,beta,limit-1,not vertical,node)
            s_branches+=res_branches
            if res_v < v_value:
                v_value = res_v
                this_move = node
            if alpha >= v_value:
                return (this_move,v_value,s_branches)
            beta = min(v_value,beta)
        return(this_move,v_value,s_branches)

