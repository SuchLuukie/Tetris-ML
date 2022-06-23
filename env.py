# Import libraries
from PIL import Image, ImageDraw
from gym.spaces import Discrete
from random import randint
import numpy as np

# Import shapes and their orientation
from shapes import shapes

class TetrisEnv:
    def __init__(self):
        # Actions:
        self.actions = {
            0: self.move_left,
            1: self.move_right,
            2: self.rotate,
            3: self.do_nothing
        }
        # Define actions space
        self.action_space = Discrete(len(self.actions))

        # Define env variables
        self.board_width = 10
        self.board_height = 20
        self.max_game_length = 1000

        # Rendering variables
        self.render_cell_size = 20
        self.background_color = (17,17,17)
        self.shape_color = (124, 252, 0)

        # Define rewards (Integers for amount of lines cleared)
        self.rewards = {
            0: 0,
            1: 4,
            2: 10,
            3: 30,
            4: 120,
            "game_over": float("-inf")
        }



    # Main function that plays the env game
    def step(self, action):
        # Default variables
        reward = 0

        # Check if we need to end game because of max game step
        if self.game_step == self.max_game_length:
            return self.board, reward, True


        # Check if there's currently an active shape
        if not self.shape.active:
            # Spawn new shape
            self.shape = Shape(self.board_width, self.board_height)

            # Add to board
            self.add_to_board(self.shape.get_shape_positions(self.shape.pos))

            # Check for game over
            if self.check_if_colliding(self.shape.pos, no_previous=True):
                reward += self.rewards["game_over"]
                return self.board, reward, True


        # Execute action if shape is still active
        if self.shape.active:
            self.actions[action]()


        # Move shape down if shape is still active
        if self.shape.active:
            new_pos = self.shape.pos
            new_pos = [new_pos[0]+1, new_pos[1]]

            # Check if it collides when moving down
            if not self.check_if_colliding(new_pos):
                # If not then remove last position's shape off of the board
                self.remove_from_board(self.shape.get_shape_positions(self.shape.pos))

                # Update shape's pos
                self.shape.pos = new_pos

                # Add new position's shape to the board
                self.add_to_board(self.shape.get_shape_positions(self.shape.pos))

            # If it does collide
            else:
                # Set shape inactive
                self.shape.active = False


        # Check lineclears and calculate reward
        if not self.shape.active:
            cleared_lines = 0
            for idx, line in enumerate(self.board):
                # If the line does not contain a 0 then it's a full line
                if not 0 in line:
                    print("[!] Cleared line")
                    self.board = np.delete(self.board, idx)
                    self.board = np.insert(self.board, 0, np.zeros(self.board_width))
                    cleared_lines += 1

            # Get the reward and add it
            reward += self.rewards[cleared_lines]

        # Add this board to history
        self.history.append(self.board.copy())

        # Increase game step
        self.game_step += 1

        # Return the state, reward and done is False
        return self.board, reward, False

    # Reset the env
    def reset(self):
        # Create the board
        self.board = np.zeros((self.board_height, self.board_width))

        # Get the first piece and place it on the board
        self.shape = Shape(self.board_width, self.board_height)
        self.add_to_board(self.shape.get_shape_positions(self.shape.pos))

        # Reset history (Used for rendering the game)
        self.history = [self.board]

        # Reset game step (Used to end game if it goes on too long)
        self.game_step = 0

    # Render the game as gif
    def render(self, filename = "game.gif"):
        gif = []

        for board in self.history:
            # Create main image and draw on it
            image = Image.new("RGB", (len(board[0]) * self.render_cell_size, len(board) * self.render_cell_size), self.background_color)
            draw = ImageDraw.Draw(image)

            # If the cell on the board is 1 then colour it 
            for y, row in enumerate(board):
                for x, cell in enumerate(row):
                    if cell != 0:
                        ny, nx = y*self.render_cell_size, x*self.render_cell_size

                        # Get the shape for the rectangle and draw it
                        shape = [nx, ny, nx + self.render_cell_size, ny + self.render_cell_size]
                        draw.rectangle(shape, fill=self.shape_color)
            
            gif.append(image.copy())

        gif[0].save(filename, save_all=True, append_images=gif[1:], loop=0)

    # Check if new position with optional new orientation collides with the old position/edges of the board
    def check_if_colliding(self, new_pos, new_orientation = False, no_previous = False):
        # Get all the positions of the new position and remove all old positions
        old = self.shape.get_shape_positions(self.shape.pos)

        if new_orientation != False:
            self.shape.orientation = new_orientation

        new = self.shape.get_shape_positions(new_pos)

        if not no_previous:
            for pos in old:
                if pos in new:
                    new.remove(pos)
        
        for pos in new:
            y, x = pos
            try:
                # Check if position is inside the board
                if y < 0 or y >= self.board_height or x < 0 or x >= self.board_width:
                    return True

                # If the cell is already taken
                elif self.board[pos[0]][pos[1]] != 0:
                    return True

            except IndexError:
                return True

        return False

    # Remove positions from board (Set to 0)
    def remove_from_board(self, positions):
        for pos in positions:
            y, x = pos
            self.board[y][x] = 0

    # Add position to board (Set to 1)
    def add_to_board(self, positions):
        for pos in positions:
            y, x = pos
            self.board[y][x] = 1

    # Function to move the active piece to the left
    def move_left(self):
        new_pos = [self.shape.pos[0], self.shape.pos[1] -1]

        if not self.check_if_colliding(new_pos):
            # Remove old positions
            self.remove_from_board(self.shape.get_shape_positions(self.shape.pos))

            # Update pos
            self.shape.pos = new_pos

            # Add new positions
            self.add_to_board(self.shape.get_shape_positions(self.shape.pos))

    # Function to move the active piece to the right
    def move_right(self):
        new_pos = [self.shape.pos[0], self.shape.pos[1] +1]

        if not self.check_if_colliding(new_pos):
            # Remove old positions
            self.remove_from_board(self.shape.get_shape_positions(self.shape.pos))

            # Update pos
            self.shape.pos = new_pos

            # Add new positions
            self.add_to_board(self.shape.get_shape_positions(self.shape.pos))

    # Function to rotate the active piece
    def rotate(self):
        old_positions = self.shape.get_shape_positions(self.shape.pos)
        original_orientation = self.shape.orientation

        # Get the next orientation
        if self.shape.orientation == len(shapes[self.shape.shape]) -1:
            new_orientation = 0

        else:
            new_orientation = self.shape.orientation + 1

        # Check if can rotate
        if not self.check_if_colliding(self.shape.pos, new_orientation=new_orientation):
            # Can rotate

            # Set new orientation
            self.shape.orientation = new_orientation

            # Remove from board
            self.remove_from_board(old_positions)

            # Get new positions
            new_positions = self.shape.get_shape_positions(self.shape.pos)

            # Add to board
            self.add_to_board(new_positions)

        else:
            # If we can't rotate
            # Set back orientation to original
            self.shape.orientation = original_orientation

    # Filler function to do nothing as action
    def do_nothing(self):
        return


# Class that holds all tetris shapes and their orientation
class Shape:
    def __init__(self, board_width, board_height):
        self.board_width, self.board_height = board_width, board_height

        # Set initial shape position
        self.pos = [0, round(self.board_width/2)-1]

        # Get a random shape
        self.shape = randint(0, len(shapes)-1)

        # Default orientation
        self.orientation = 0

        # Boolean to see if still active
        self.active = True


    def get_shape_positions(self, pos):
        relative = shapes[self.shape][self.orientation]
        absolute = [[shape[0] + pos[0], shape[1] + pos[1]] for shape in relative]
        return absolute


# Temp
TetrisEnv()