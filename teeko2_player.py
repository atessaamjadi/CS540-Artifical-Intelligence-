import random
import copy
import math
import timeit

class Teeko2Player:
    """ An object representation for an AI game player for the game Teeko2.
    """
    board = [[' ' for j in range(5)] for i in range(5)]
    pieces = ['b', 'r']

    def __init__(self):
        """ Initializes a Teeko2Player object by randomly selecting red or black as its
        piece color.
        """
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]

    def make_move(self, state):
        """ Selects a (row, col) space for the next move. You may assume that whenever
        this function is called, it is this player's turn to move.

        Args:
            state (list of lists): should be the current state of the game as saved in
                this Teeko2Player object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method (use
                place_piece() instead). Any modifications (e.g. to generate successors)
                should be done on a deep copy of the state.

                In the "drop phase", the state will contain less than 8 elements which
                are not ' ' (a single space character).

        Return:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

        Note that without drop phase behavior, the AI will just keep placing new markers
            and will eventually take over the board. This is not a valid strategy and
            will earn you no points.
        """

     
        # TODO: detect drop phase
        drop_phase = self.detectDropPhase(state)
 
        #generate all succ's
        succList = []
        succList = self.succ(state, drop_phase)

        winningSucc = None

        minSucc = []
        
        #calculate game value of each successor
        for succ in succList: 
            if self.game_value(succ) == 1:
                winningSucc = succ
                break
            minValue = self.Min_Val(state, 0, 3, -math.inf, math.inf)
            minSucc.append((succ, minValue))
    
        #get which succ has the highest min value
        succState = None
        temp = None
        if winningSucc is not None:
            succState = winningSucc
        else:
            temp = minSucc[0][1]
            succState = minSucc[0][0]
            for i in minSucc:
                if temp < i[1]:
                    temp = i[1]
                    succState = i[0]

        move = []

        #figure out where to place the next move
        for row in range(5):
            for col in range(5):
                #compare succ and state
                if drop_phase:
                    if succState[row][col] != state[row][col]:
                        move.append((row, col)) 
                        return move                    
                
                else:
                    #return a list of two tuples, 0 = new position 1 = old position
                    if succState[row][col] != state[row][col]:
                        #new position
                        if state[row][col] == " ":
                            move.insert(0, (row , col))
                        #old position
                        else:
                            move.append((row,col))                     
                      
        return move 

        '''#if not drop_phase:
            # TODO: choose a piece to move and remove it from the board
            # (You may move this condition anywhere, just be sure to handle it)
            #
            # Until this part is implemented and the move list is updated
            # accordingly, the AI will not follow the rules after the drop phase!
 
        # select an unoccupied space randomly

        # TODO: implement a minimax algorithm to play better
        move = []
        (row, col) = (random.randint(0,4), random.randint(0,4))
        while not state[row][col] == ' ':
            (row, col) = (random.randint(0,4), random.randint(0,4))
            
        

        # ensure the destination (row,col) tuple is at the beginning of the move list
        move.insert(0, (row, col))
        return move

        '''
    #detect drop phase
    def detectDropPhase(self, state):
        numPieces = 0
        for i in range(len(state)):
            for j in range(len(state[i])):
                if(state[i][j] != ' '):
                    numPieces = numPieces + 1

        if numPieces == 8:
            return False
        
        return True

    #using alpha-beta pruning
    def Min_Val(self, state, depthOfState, finalDepth, alpha, beta):

        if self.game_value(state) != 0:
            return self.game_value(state)

        if finalDepth == depthOfState:
            return self.heuristic_game_value(state)
        
        succ2 = self.succ(state, self.detectDropPhase(state))

        for i in succ2:
            temp = self.Max_Value(i, depthOfState + 1, finalDepth, alpha, beta)
            beta = min(beta, temp)
            
            if alpha >= beta:
                return beta
    
        return alpha

    #using alpha-beta pruning
    def Max_Value(self, state, depthOfState, finalDepth, alpha, beta):

        if self.game_value(state) != 0:
            return self.game_value(state)

        if finalDepth == depthOfState:
            return self.heuristic_game_value(state)   

        succ2 = self.succ(state, self.detectDropPhase(state))

        for i in succ2:
            temp = self.Min_Val(i, depthOfState + 1, finalDepth, alpha, beta)
            alpha = max(alpha, temp)
            
            if alpha >= beta:
                return beta
        
        return alpha  

    #heuristic if not terminal state
    def heuristic_game_value(self, state): 
        h = self.game_value(state)
        if h != 0:
            return h
        else:
            max = []
            min = []
        for i in range(5):
            for j in range(5):
                if state[i][j] == self.my_piece:
                    max.append([i,j])
                if state[i][j] == self.opp:
                    min.append([i,j])
        #get average max distance of each token
        max_values = []
        for i in range(len(max)):
            for j in range(len(max)):
                if j <= i:
                    continue
                x = max[j][0] - max[i][0]
                x = math.pow(x,2)
                y = max[j][1] - max[i][1]
                y = math.pow(y,2)
                max_values.append(math.sqrt(x + y))
        maxTot = 0
        if len(max_values) < 1:
            maxTot = 2
        else :
            for k in max_values:
                maxTot = maxTot + k
            maxTot = maxTot / max_values.__len__()
        
        #get average min distance
        min_values = []
        for i in range(len(min)):
            for j in range(len(min)):
                if j <= i:
                    continue
                x = min[j][0] - min[i][0]
                x = math.pow(x,2)
                y = min[j][1] - min[i][1]
                y = math.pow(y,2)
                min_values.append(math.sqrt(x + y))
        minTot = 0
        if len(min_values) < 1:
            minTot = 2
        else:
            for k in min_values:
                minTot = minTot + k
            minTot = minTot / min_values.__len__()

        # get approx normalization of avg distances
        maxTot = 1 / maxTot
        minTot = 1 / minTot
        if maxTot >= 1:
            maxTot = 0.99
        if minTot >= 1:
            minTot = 0.99
        
        if maxTot >= minTot:
            return maxTot
        else:
            return (-1 * minTot)

    # takes in a board state and returns a list of the legal successors. 
    def succ(self, state, drop_phase):
        succ = []
        # during the drop phase, this simply means adding a new piece of the current player's type to the board;
        if drop_phase:
            for i in range(5):
                for j in range(5):
                    if state[i][j] == ' ':
                        copyState = copy.deepcopy(state)
                        copyState[i][j] = self.my_piece
                        succ.append(copyState)

        # during continued gameplay, this means moving any one of the current player's pieces to an unoccupied location on the board, 
        # adjacent to that piece.
        else: 
            for i in range(5):
                for j in range(5):
                    if state[i][j] == self.my_piece:
                        #conditions for moving adjacent
                        #check if 8 if statements are in bounds
                        #check moving up down & left right
                        if i+1 < len(state):
                            if state[i+1][j] == ' ':
                                copyState = copy.deepcopy(state)
                                copyState[i+1][j] = self.my_piece
                                copyState[i][j] = ' '
                                succ.append(copyState)

                        if i-1 >= 0:
                            if state[i-1][j] == ' ':
                                copyState = copy.deepcopy(state)
                                copyState[i-1][j] = self.my_piece
                                copyState[i][j] = ' '
                                succ.append(copyState)

                        if j+1 < len(state):
                            if state[i][j+1] == ' ':
                                copyState = copy.deepcopy(state)
                                copyState[i][j+1] = self.my_piece
                                copyState[i][j] = ' '
                                succ.append(copyState)
                        
                        if j-1 >= 0:
                            if state[i][j-1] == ' ':
                                copyState = copy.deepcopy(state)
                                copyState[i][j-1] = self.my_piece
                                copyState[i][j] = ' '
                                succ.append(copyState)

                        #check diagonal
                        if i-1 >= 0 and j-1 >= 0:
                            if state[i-1][j-1] == ' ':
                                copyState = copy.deepcopy(state)
                                copyState[i-1][j-1] = self.my_piece
                                copyState[i][j] = ' '
                                succ.append(copyState)

                        if i-1 >=0 and j+1 < len(state):
                            if state[i-1][j+1] == ' ':
                                copyState = copy.deepcopy(state)
                                copyState[i-1][j+1] = self.my_piece
                                copyState[i][j] = ' '
                                succ.append(copyState)

                        if i+1 < len(state) and j+1 < len(state):
                            if state[i+1][j+1] == ' ':
                                copyState = copy.deepcopy(state)
                                copyState[i+1][j+1] = self.my_piece
                                copyState[i][j] = ' '
                                succ.append(copyState)

                        if i+1 < len(state) and j-1 >= 0:
                            if state[i+1][j-1] == ' ':
                                copyState = copy.deepcopy(state)
                                copyState[i-1][j-1] = self.my_piece
                                copyState[i][j] = ' '
                                succ.append(copyState)
        return succ

    def opponent_move(self, move):
        """ Validates the opponent's next move against the internal board representation.
        You don't need to touch this code.

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)

    def place_piece(self, move, piece):
        """ Modifies the board representation using the specified move and piece

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row)+": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    def game_value(self, state):
        """ Checks the current board status for a win condition

        Args:
        state (list of lists): either the current state of the game as saved in
            this Teeko2Player object, or a generated successor state.

        Returns:
            int: 1 if this Teeko2Player wins, -1 if the opponent wins, 0 if no winner

        TODO: complete checks for diagonal and diamond wins
        """

        #print("now we are at game value")

        # check horizontal wins
        for row in state:
            for i in range(2):
                if row[i] != ' ' and row[i] == row[i+1] == row[i+2] == row[i+3]:
                    return 1 if row[i]==self.my_piece else -1

        # check vertical wins
        for col in range(5):
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i+1][col] == state[i+2][col] == state[i+3][col]:
                    return 1 if state[i][col]==self.my_piece else -1
    
        # TODO: check \ diagonal wins
        if state[1][0] != ' ' and state[1][0] == state[2][1] == state[3][2] == state[4][3]:
            return 1 if state[1][0] == self.my_piece else -1
        if state[0][0] != ' ' and state[0][0] == state[1][1] == state[2][2] == state[3][3]:
            return 1 if state[0][0] == self.my_piece else -1
        if state[1][1] != ' ' and state[1][1] == state[2][2] == state[3][3] == state[4][4]:
            return 1 if state[1][1] == self.my_piece else -1
        if state[0][1] != ' ' and state[0][1] == state[1][2] == state[2][3] == state[3][4]:
            return 1 if state[0][1] == self.my_piece else -1

        
        # TODO: check / diagonal wins
        if state[0][3] != ' ' and  state[0][3] == state[1][2] == state[2][1] == state[3][0]:
            return 1 if state[0][3] == self.my_piece else -1
        if state[0][4] != ' ' and state[0][4] == state[1][3] == state[2][2] == state[3][1]:
            return 1 if state[0][4] == self.my_piece else -1
        if state[1][3] != ' ' and state[1][3] == state[2][2] == state[3][1] == state[4][0]:
            return 1 if state[1][3] == self.my_piece else -1
        if state[1][4] != ' ' and state[1][4] == state[2][3] == state[3][2] == state[4][1]:
            return 1 if state[1][4] == self.my_piece else -1
    
        # TODO: check diamond wins
        for row in range(1,4):
            for col in range(1,4):
                if state[row][col] == ' ' and state[row+1][col] != ' ' and state[row+1][col] == state[row-1][col] == state[row][col+1] == state[row][col-1]:
                    return 1 if state[row+1][col] == self.my_piece else -1

        return 0 # no winner yet

############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################
def main():
    print('Hello, this is Samaritan')
    ai = Teeko2Player()
    piece_count = 0
    turn = 0

    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved at "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved from "+chr(move[1][1]+ord("A"))+str(move[1][0]))
            print("  to "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0])-ord("A")),
                                    (int(move_from[1]), ord(move_from[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")


if __name__ == "__main__":
    main()
