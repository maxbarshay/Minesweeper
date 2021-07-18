# Name:         Max Barshay
# Course:       CSC 480
# Instructor:   Daniel Kauffman
# Assignment:   Mine Shafted
# Term:         Summer 2020

from typing import Generator, List, Tuple, Set, Dict, Optional
import itertools



class BoardManager:

    def __init__(self, board: List[List[int]]):
        """
        An instance of BoardManager has two attributes.

            size: A 2-tuple containing the number of rows and columns,
                  respectively, in the game board.
            move: A callable that takes an integer as its only argument to be
                  used as the index to explore on the board. If the value at
                  that index is a clue (non-mine), this clue is returned;
                  otherwise, an error is raised.

        This constructor should only be called once per game.

        >>> board = [[0, 1, 1], [0, 2, -1], [0, 2, -1], [0, 1, 1]]
        >>> bm = BoardManager(board)
        >>> bm.size
        (4, 3)
        >>> bm.move(4)
        2
        >>> bm.move(5)
        Traceback (most recent call last):
        ...
        RuntimeError
        """
        self.size: Tuple[int, int] = (len(board), len(board[0]))
        it: Generator[int, int, None] = BoardManager._move(board, self.size[1])
        next(it)
        self.move: Callable[[int], int] = it.send

    @staticmethod
    def _move(board: List[List[int]], width: int) -> Generator[int, int, None]:
        """
        A generator that may be sent integers (indices to explore on the board)
        and yields integers (clues for the explored indices).

        Do not call this method directly; instead, call the |move| instance
        attribute, which sends its index argument to this generator.
        """
        index = (yield 0)
        while True:
            clue = board[index // width][index % width]
            if clue == -1:
                raise RuntimeError
            index = (yield clue)


class Clue:

    def __init__(self, index: int, indices: Tuple[int, int], 
            adj_unexp_squares: List[int], 
            domain: List[Tuple[bool, ...]], clue_value: int):

        self.index = index
        self.indices = indices
        self.adj_unexp_squares = adj_unexp_squares
        self.domain = domain
        self.clue_value = clue_value


    def __repr__(self):

        return "Index: >" + str(self.index) \
         + "< -- Domain: " + str(self.domain)
#
    # + " -- Indices: " + str(self.indices) 
            # + " -- Clue Value: " + str(self.clue_value) 
            # + \#+ "!! -- Adj Unexplored Squares: " + \
        #str(self.adj_unexp_squares) 


class Solver:

    def __init__(self, bm: BoardManager, clues: Dict[int, Clue]):

        self.bm = bm
        self.board_size = bm.size
        self.width = bm.size[1]
        self.my_field: List[List[int]] = \
         [[None for i in range(self.board_size[1])] for j 
                in range(self.board_size[0])]
        self.uncovered_squares: List[int] = []
        self.clues = clues

        # note that these unmodified clues get wiped after every round


    def index_converter(self, index: int) -> Tuple[int, int]:
        """
        Takes in the index on the game board and converts it 
        to a tuple containing the two indexes into the 2D List.
        """
        #print((index // dimension[0], index % dimension[1]))
        # check validity of this
        return (index // self.width, index % self.width)


    def get_adj_squares(self, index: int) -> List[int]:
        """
        Given the dimension of the game board and the index. 
        Find all of the adjacent squares to the index. 
        """
        width = self.width
        adj_squares: List[int] = []
        if not (index % width == 0 or index % width == width - 1): 
        # if in middle 
            adj_squares.append(index + 1)
            adj_squares.append(index - 1)
            adj_squares.append(index + width)
            adj_squares.append(index - width)
            adj_squares.append(index - width - 1)
            adj_squares.append(index - width + 1)
            adj_squares.append(index + width + 1)
            adj_squares.append(index + width - 1)
        elif index % width == 0: # if on left edge
            adj_squares.append(index + 1)
            adj_squares.append(index + width)
            adj_squares.append(index - width)
            adj_squares.append(index - width + 1)
            adj_squares.append(index + width + 1)
        elif index % width == width - 1: # if on right edge
            adj_squares.append(index - 1)
            adj_squares.append(index + width)
            adj_squares.append(index - width)
            adj_squares.append(index - width - 1)
            adj_squares.append(index + width - 1)
        adj_squares = [sq for sq in adj_squares if sq < \
                (self.bm.size[0] * self.bm.size[1])]
        return adj_squares


    def get_relevant_adj_squares(self, index: int) -> List[int]:
        """
        Goes from all adjacent squares to just the relevant ones.
        """
        adj_squares: List[int] = self.get_adj_squares(index)
        filtered_adj_squares: List[Clue] = []
        for square in adj_squares:
            if square >= 0:
                indices = self.index_converter(square)
                if self.my_field[indices[0]][indices[1]] is None:
                    filtered_adj_squares.append(square)
        return filtered_adj_squares


    def find_frontier_clues(self) -> List[Clue]:
        """
        Find's the indices of all explored squares that have at 
        least one adjacent unexplored square.
        """
        clues: List[Clue] = []
        num_squares = self.bm.size[0] * self.bm.size[1]
        for i in range(num_squares):
            indices = self.index_converter(i)
            if self.my_field[indices[0]][indices[1]] is not None:
                rel_adj_squares = self.get_relevant_adj_squares(i)
                clue_value = self.my_field[indices[0]][indices[1]]
                if len(rel_adj_squares) is not None:
                    clues.append(Clue(i, indices, 
                            sorted(rel_adj_squares), None, clue_value))
        return clues


    def add_domains(self, clues: List[Clue]) -> None:
        """
        Given a clue, this function finds the domain of 
        all possible clues. This function will be using 
        itertools permutations function.
        """
        for clue in clues:
            clue_domain = Solver.make_permutations(clue.adj_unexp_squares, 
                    clue.clue_value)
            clue.domain = clue_domain



    @staticmethod
    def make_permutations(adj_squares: List[int], clue_value: int) \
         -> List[Tuple[bool, ...]]:
        """
        Static method of the solver class that makes permutations.
        """
        length = len(adj_squares)
        num_falses = length - clue_value
        trues = [1] * clue_value
        falses = [-1] * num_falses
        to_permute = trues + falses
        permutations = list(map(list, itertools.permutations(to_permute)))
        for i in range(len(permutations)):

            for j in range(len(permutations[i])):

                permutations[i][j] = adj_squares[j] * permutations[i][j]

        permutations = list(set(map(tuple, permutations)))
    
        return permutations




    def find_all_arcs(self, clues: List[Clue]) -> Set[Tuple[Clue, Clue]]:
        """
        Used at very beginning of minesweeping game to find all arcs
        and place those arcs into a list. Each arc is just a tuple of
        the two indexes that make it up.
        """
        arcs: Set[Tuple[Clue, Clue]] = []
        for i in range(len(clues) - 1):
            for j in range(i, len(clues)):
                if i != j: 
                    if len(set(clues[i].adj_unexp_squares).\
                        intersection(clues[j].adj_unexp_squares)) != 0:
                        arcs.append((clues[i], clues[j]))
                        arcs.append((clues[j], clues[i]))
        return arcs



    def initialize_arcs(self) -> Set[Tuple[Clue, Clue]]:
        """
        A helper function to combine the steps of finding 
        frontier clues and adding the domains.
        """
        # self.clues = []
        frontier = self.find_frontier_clues()
        self.add_domains(frontier)
        self.clues = self.get_clue_dict(frontier)
        arcs = self.find_all_arcs(frontier)
        return arcs

    def get_clue_dict(self, clues: List[Clue]) -> Dict[int, Clue]:
        """
        Upon initialization this method, creates a dictionary containing
        as keys, the indexes of the clues and as values the clue objects 
        themselves.
        """
        clue_dict = {}
        for clue in clues:
            clue_dict[clue.index] = clue
        return clue_dict


    def uncover_square(self, index: int) -> None:
        """
        This both uncovers a square as well as updates my_field which is 
        keeping track of the mines that I have swept so far.
        """
        clue = self.bm.move(index)
        self.uncovered_squares.append(index)
        new_index = self.index_converter(index)
        self.my_field[new_index[0]][new_index[1]] = clue



    def revise(self, arc: Tuple[Clue, Clue]) -> bool:
        """
        This function will take in two Clues.
        Xi: The first Clue in the arc. 
        The Clue whos domain may potentially be modified.
        Xj: The second Clue in the arc. 
        This Clue's domain will not be changed.
        """
        revised = False
        di = arc[0].domain
        dj = arc[1].domain
        to_remove = []
        for x in di:
            if not self.is_compatable(x, dj):
                to_remove.append(x)
                revised = True
        arc[0].domain = [arc for arc in arc[0].domain if arc not in to_remove]
        return revised

    def is_compatable(self, x, dj) -> bool:
        """
        This method takes in one arrangement in the domain of Xi, namely x, and
        sees if it is compatible with any y in Dj.
        If it is not then we return False.
        """
        compatable = True
        pos_x = [abs(val) for val in x]
        mixed_y = dj[0]
        pos_y = [abs(val) for val in mixed_y]
        abs_intersect = len(set(pos_x).intersection(pos_y))

        for j in range(len(dj)):

            intersection = set(x).intersection(dj[j])

            if len(intersection) >= abs_intersect:

                return compatable

        return False


    def at_least_one_neg(self, domain: List[Tuple[int, ...]]) -> bool:
        """
        Given a domain of length one, this method checks to 
        see whether at least one element in the domain is negative.
        """
        one_neg = False
        assignment = domain[0]
        for i in range(len(assignment)):
            if assignment[i] < 0:
                # print(assignment[i])
                return True
        return one_neg





    def check_single_domain(self, arc) -> Optional[List[Tuple[int, ...]]]:
        """
        This method checks if any Clue in an arc has a domain of just one.
        If this is the case that means we can uncover that 
        square and reset the puzzle.
        Returns the index of the square that we can uncover. 
        """
        xi = arc[0]
        xj = arc[1]
        if len(xi.domain) == 1 and len(xi.domain[0]) != 0:
            if self.at_least_one_neg(xi.domain):
                return xi.domain
        elif len(xj.domain) == 1 and len(xj.domain[0]) != 0:
            if self.at_least_one_neg(xj.domain):
                return xj.domain


    def smart_determine_uncover(self, domain: List[Tuple[int, ...]]) -> int:
        """
        Given a domain of length 1, this method determines 
        which square is safe to uncover.
        It returns the index of the square that can be uncovered. 
        """
        assignment = domain[0]
        for i in range(len(assignment)):
            if assignment[i] < 0:
                return abs(assignment[i])


    def all_agree(self) -> List[int]:
        """
        Once all possible arcs have been eliminated, we call this function
        to see if every since squares domain agrees on anything. If they all
        agree one square is safe. We can uncover that square. 
        """
        keys = self.clues.keys()
        unique_squares = []
        for key in keys:
            for i in range(len(self.clues[key].domain)):
                for j in range(len(self.clues[key].domain[i])):
                    if abs(self.clues[key].domain[i][j]) not in unique_squares:
                        unique_squares.append(abs(self.clues[key].domain[i][j]))
        res: List[int] = []
        for num in unique_squares:
            seen_pos = False
            for key in keys:
                for i in range(len(self.clues[key].domain)):
                    for j in range(len(self.clues[key].domain[i])):
                        if len(self.clues[key].domain[i]) > 0:
                            if self.clues[key].domain[i][j] == num:
                                seen_pos = True
            if not seen_pos:
                res.append(num)
        return res


    def ac_3_algorithm(self) -> None:
        """
        This runs the AC-3 Algorithm. Using a set instead 
        of the traditional queue to store each arc because I want to ensure 
        that I do not put duplicate arcs into the set.
        """
        # note that set.pop() removes an arbitrary element
        # note that set.add(element) adds an element to the set
        while True:

            arcs = self.initialize_arcs()

            while len(arcs) != 0:
                
                arc = arcs.pop() # remove an arbitrary element

                single = self.check_single_domain(arc)
                
                if single is not None and len(single) != 0:
                    to_uncover = self.smart_determine_uncover(single)

                    self.uncover_square(to_uncover)

                    break

                if self.revise(arc):

                    xi_neighbors = self.get_adj_squares(arc[0].index)

                    for xk in self.clues.keys():
                        if xk in xi_neighbors \
                            and xk != arc[1].index:
                            new_arc = (self.clues[xk], arc[0])
                            arcs.append(new_arc)

            if len(arcs) == 0:

                can_remove = self.all_agree()

                if can_remove != []:

                    self.uncover_square(can_remove[0])

                else:

                    for i in range(self.bm.size[0]):

                        for j in range(self.bm.size[1]):

                            if self.my_field[i][j] is None:

                                self.my_field[i][j] = -1

                    return


def sweep_mines(bm: BoardManager) -> List[List[int]]:
    """
    Given a BoardManager (bm) instance, return a solved board (represented as a
    2D list) by repeatedly calling bm.move(index) until all safe indices have
    been explored. If at any time a move is attempted on a non-safe index, the
    BoardManager will raise an error; this error signifies the end of the game
    and should not attempt to be caught.

    >>> board = [[0, 1, 1], [0, 2, -1], [0, 2, -1], [0, 1, 1]]
    >>> bm = BoardManager(board)
    >>> sweep_mines(bm)
    [[0, 1, 1], [0, 2, -1], [0, 2, -1], [0, 1, 1]]
    """
    solver = Solver(bm, None)
    solver.uncover_square(0)
    if bm.size[1] > 1 and bm.size[0] > 1:
        solver.uncover_square(1)
        solver.uncover_square(0 + solver.board_size[1])
        solver.uncover_square(0 + solver.board_size[1] + 1)
    elif bm.size[1] > 1:
        solver.uncover_square(1)
    elif bm.size[0] > 1:
        solver.uncover_square(0 + solver.board_size[1])

    solver.ac_3_algorithm()

    return solver.my_field





def main() -> None:  

    board = [[0,1,-1,1],
              [1,2,1,1],
              [-1,1,0,0],
              [1,1,0,0]]

    # optionally uncomment these additional boards

    # board2 = [[0,0,0,1,1,1,1,1,1],
    #           [0,0,0,1,-1,1,1,-1,2],
    #           [0,0,0,1,2,2,2,2,-1],
    #           [0,0,1,1,2,-1,1,1,1],
    #           [0,0,1,-1,2,2,2,1,0],
    #           [1,1,2,1,1,2,-1,2,0],
    #           [2,-1,2,0,0,3,-1,3,0],
    #           [2,-1,2,0,0,2,-1,2,0],
    #           [1,1,1,0,0,1,1,1,0]]

    # board3 = [[0,0,0,0,1,1,1,0],
    #       [1,1,1,0,1,-1,1,0],
    #       [1,-1,1,0,1,1,2,1],
    #       [1,1,1,0,0,0,2,-1],
    #       [0,0,0,0,0,0,2,-1]]



    # board4 = [[0,0,0,0,0,0,0,0,0],
    #           [0,0,0,1,2,2,2,1,1],
    #           [0,0,1,2,-1,-1,2,-1,1],
    #           [1,1,1,-1,3,2,2,1,1],
    #           [-1,1,2,2,2,0,0,0,0],
    #           [2,2,1,-1,1,0,0,0,0],
    #           [-1,2,2,2,1,0,0,0,0],
    #           [2,3,-1,1,0,0,0,1,1],
    #           [-1,2,1,1,0,0,0,1,-1]]

    # this is the general pattern to run the code

    bm = BoardManager(board)

    solved = sweep_mines(bm)

    print(solved)

    assert solved == board
    


if __name__ == "__main__":
    main()
