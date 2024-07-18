"""
Code for the game: Translation from Java to Python (Powered by ChatGPT)
"""


class QuentinGame:
    
    def __init__(self, size):
        self.size = size        #side of the board
        self.board = [-1] * (size * size)       #board initialization

    def __str__(self):
        rows = "abcdefghijklm"
        printed = "   " + " ".join(f"{i:2}" for i in range(self.size)) + "\n"
        for row in range(self.size):
            line = f"{rows[row]} "
            for col in range(self.size):
                value = self.board[row * self.size + col]
                if value == 0:
                    line += " B"
                elif value == 1:
                    line += " W"
                else:
                    line += " ."
            printed += line + "\n"
        return printed   


    def neighbours(self, idx, exclude):
        """
        returns the orthogonally adjacent points to the input point
        """
        mylist = []
        if idx + self.size < len(self.board) and idx + self.size not in exclude:
            mylist.append(idx + self.size)
        if idx + 1 < len(self.board) and idx // self.size == (idx + 1) // self.size and idx + 1 not in exclude:
            mylist.append(idx + 1)
        if idx - self.size >= 0 and idx - self.size not in exclude:
            mylist.append(idx - self.size)
        if idx - 1 >= 0 and idx // self.size == (idx - 1) // self.size and idx - 1 not in exclude:
            mylist.append(idx - 1)
        return mylist


    def diagonals(self, idx):
        """
        returns the indexes of the diagonal points of the input point
        """
        mylist = []
        if idx + self.size + 1 < len(self.board) and (idx + 1) // self.size == idx // self.size:
            mylist.append(idx + self.size + 1)
        if idx + self.size - 1 < len(self.board) and (idx - 1) // self.size == idx // self.size and idx != 0:
            mylist.append(idx + self.size - 1)
        if idx - self.size - 1 >= 0 and (idx - 1) // self.size == idx // self.size:
            mylist.append(idx - self.size - 1)
        if idx - self.size + 1 >= 0 and (idx + 1) // self.size == idx // self.size:
            mylist.append(idx - self.size + 1)
        return mylist


    def legal_move(self, last_move):
        """
        a move is legal if at the end of the player's turn there are NOT any couples of points with the same color s.t.
        they are diagonally adjacent and the do not share any orthogonally adjacent, like-colored neighbor 
        """
        like_colored_diags = 0
        like_colored_neighbours = []
        ## check for (and store the locations of) the like-colored neighbours of the last point
        for i in self.neighbours(last_move, []):
            if self.board[i] == self.board[last_move]:
                like_colored_neighbours.append(i)
        ## verify if any of the diagonally adjacent points are like-colored and share any of the neighbours from before
        for i in self.diagonals(last_move):
            if self.board[i] == self.board[last_move]:
                like_colored_diags = like_colored_diags + 1
                common_neighbours = list(set(self.neighbours(i, [])) & set(like_colored_neighbours))
                # if so, the move is legal!
                if len(common_neighbours) > 0:
                    return True        
        ##if you find at least one like-coloured diagonal location, but s.t. the shared adjacent locations are not like-coloured --> ILLEGAL MOVE
        if like_colored_diags > 0:
            return False        
        return True


    def adiacent_location(self, region, idx):
        """
        determine if a certain empty point in the board is adjacent to any of the points of a certain region 
        (either directly or indirectly, i.e., through other points)
        """
        ## find the empty neighbours of a starting region
        region_neighbours = set()
        for curr in region:
            region_neighbours.update(self.neighbours(curr, []))
            region_neighbours.difference_update([i for i in region_neighbours if self.board[i] != -1])
        
        ## find the neighbours of the point of which you want to verify the adjacency to the region
        extended_idx_neighbours = set(self.neighbours(idx, []))
        extended_idx_neighbours.difference_update([i for i in extended_idx_neighbours if self.board[i] != -1])
        extended_idx_neighbours.add(idx)
        
        ## if the two sets of neighbours overlap, the point is adjacent to the region 
        temp_set = extended_idx_neighbours.intersection(region_neighbours)
        if temp_set:
            return True
        
        #otherwise, progressively expand the set of neighbours of the point, until...
        while True:
            for x in extended_idx_neighbours:
                temp_set.update(self.neighbours(x, []))
                temp_set.difference_update([i for i in temp_set if self.board[i] != -1])
            # either it doesn't change anymore,
            if extended_idx_neighbours == temp_set:
                return False
            # or you finally find the intersection with the region neighbours set!
            extended_idx_neighbours = temp_set
            temp_set = extended_idx_neighbours.intersection(region_neighbours)
            if temp_set:
                return True

    
    def candidate_for_territory(self, region):
        """
        a region is a candidate to become a territory if each of its points has at least two filled neighbours 
        """
        for curr in region:
            cnt = sum(1 for i in self.neighbours(curr, []) if self.board[i] != -1)
            if cnt < 2:
                return False
        return True


    def fill_territory(self, region, last_move):
        """
        evaluate if a region (group of adjacent locations in the board) can be considered a territory:
        each location of the region should have at least two neighbours which have been filled by any player;
        if the condition is met, all the locations will get the color of the player who has the majority
        of the filled neighbours (in the event of a tie, the territory gets the color of the player who did not 
        do the last move)
        """
        ## determine the set of all the distinct neighbours of a region 
        neighbours_union = set()
        for i in region:
            neighbours_union.update(self.neighbours(i, []))
        neighbours_union.difference_update(region)
        distinct_neighbours = list(neighbours_union)
        ## evaluate the color to be assigned to the region
        cnt_black = sum(1 for i in distinct_neighbours if self.board[i] == 0)
        cnt_white = sum(1 for i in distinct_neighbours if self.board[i] == 1)
        if cnt_black == cnt_white:
            if self.board[last_move] == 1:
                cnt_black += 1
            else:
                cnt_white += 1
        ## assignment
        replacement = 0 if cnt_black > cnt_white else 1
        for index in region:
            self.board[index] = replacement

    

    def update_board(self, is_black, move, unavail=[]):
        """
        include the new move and all the related changes in the board (if they are compliant to the rules)
        """
        empty_locations = [i for i, x in enumerate(self.board) if x == -1]
        valid_move = move in empty_locations and 0 <= move < len(self.board)
        ## if the last move corresponds to a location which is already filled or one out of the board, the move is not valid
        if not valid_move:
            unavail.append(move)
            return False
        
        ## if the first validity check is passed, the second verification phase can start:
        self.board[move] = 0 if is_black else 1
        region = []
        territories = []
        ## scan through all the empty locations and detect regions in the board
        while empty_locations:
            for loc in empty_locations:
                if not region or self.adiacent_location(region, loc):
                    region.append(loc)
            ## once a certain region has been completely defined (no other locations can be inserted),
            # verify if the conditions to be considered a territory hold, and if so, fill the region accordingly 
            if self.candidate_for_territory(region):
                self.fill_territory(region, move)
                territories.extend(region)

            ## update the list of empty locations
            empty_locations = [x for x in empty_locations if x not in region]
            region.clear()
        
        ## if after filling the territories (if any), the last move is still illegal, remove the last edits to the board
        if not self.legal_move(move):
            self.board[move] = -1
            for idx in territories:
                self.board[idx] = -1
            territories.clear()
            unavail.append(move)
            return False
        return True


    def winning_path_lookup(self, idx, exclude, winning_path):
        """
        checks for a winning path for a specific player:
        a top-to-bottom way of orthogonally adjacent points, for the black player; a left-to-right way for the white player  
        """
        if idx in winning_path:
            return False
        winning_path.add(idx)
        neighbours = self.neighbours(idx, exclude)
        ## define all the potential arrival points for the player 
        arrival = list(range(self.size * (self.size - 1), self.size * self.size)) if self.board[idx] == 0 else list(range(self.size - 1, self.size * self.size, self.size))
        if idx in arrival:
            return True
        ## look for a winning path recursively starting from each like-colored neighbour of the point
        for neighbour in neighbours:
            if self.board[neighbour] == self.board[idx] and self.winning_path_lookup(neighbour, exclude, winning_path):
                return True
        winning_path.remove(idx)
        return False
    

    def gameover(self):
        """
        verifies if there is a winning path for both the players
        """
        winning_path = set()
        black_won = any(self.board[i] == 0 and self.winning_path_lookup(i, [], winning_path) for i in range(self.size))        
        if black_won:
            # print("BLACK won")
            return 0
        white_won = any(self.board[i * self.size] == 1 and self.winning_path_lookup(i * self.size, [], winning_path) for i in range(self.size))
        if white_won:
            # print("WHITE won")
            return 1
        return -1