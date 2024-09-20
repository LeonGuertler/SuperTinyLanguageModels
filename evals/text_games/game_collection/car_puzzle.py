import re, random
from evals.text_games.game_collection.game_interface import GameInterface

class CarPuzzleGame(GameInterface):
    def __init__(self, render=False):
        """
        Initialize the Car Puzzle Game.
        Args:
            render (bool): If True, prints game states for debugging.
        """
        self.name = "Car Puzzle"
        self.render = render

    def reset(self):
        """Reset the game to its initial state."""
        self.game_over = False
        self.phase = 1  # Phase 1: Player 0 provides puzzle and solution
        self.board = None
        self.exit = None
        self.cars = {}
        self.player_prompts = {
            0: self._generate_puzzle_creator_prompt(),
            1: self._generate_puzzle_solver_prompt()
        }
        return self.player_prompts

    def _generate_puzzle_creator_prompt(self):
        """
        Generate the prompt for the puzzle-creator player (Player 0).
        """
        prompt = (
            "You are the Puzzle Creator in the 'Car Puzzle' game.\n"
            "Your task is to create a car puzzle by specifying the grid size, placing cars, and providing a solution.\n"
            "There is a specific target car with ID 0 that needs to reach the exit.\n"
            "Cars are numbered uniquely and can only move in the direction they are oriented (forward and backward).\n"
            "Each car must be at least two squares long and be placed in a straight line (horizontal or vertical).\n"
            "Format your submission using the following formalized language (use square brackets for keywords):\n"
            "\n"
            "1. Define Grid Size:\n"
            "[GridSize] width height\n"
            "Example: [GridSize] 6 6\n"
            "\n"
            "2. Place Cars:\n"
            "[PlaceCar] car_id orientation x y length\n"
            " - car_id: Integer identifier for the car (0 is the target car).\n"
            " - orientation: H for horizontal, V for vertical.\n"
            " - x y: Coordinates of the starting position (top-left for H, top for V).\n"
            " - length: Length of the car (minimum 2).\n"
            "Example: [PlaceCar] 0 H 2 2 2\n"
            "\n"
            "3. Specify Exit Point (optional, default is right edge on the same row as the target car):\n"
            "[Exit] x y\n"
            "Example: [Exit] 5 2\n"
            "\n"
            "4. Provide Solution Steps:\n"
            "List of moves using the following format:\n"
            "[Move] car_id steps\n"
            " - car_id: Integer identifier of the car to move.\n"
            " - steps: Positive or negative integer indicating the number of steps to move (positive for forward, negative for backward).\n"
            "Example: [Move] 0 1\n"
            "\n"
            "Submit your puzzle and solution together.\n"
            "Full Example:\n"
            "[GridSize] 6 6\n"
            "[PlaceCar] 0 H 2 2 2\n"
            "[PlaceCar] 1 V 0 0 3\n"
            "[PlaceCar] 2 V 4 0 2\n"
            "[Exit] 5 2\n"
            "[Move] 1 1\n"
            "[Move] 0 2\n"
        )
        return prompt

    def _generate_puzzle_solver_prompt(self):
        """
        Generate the prompt for the puzzle-solving player (Player 1).
        """
        prompt = (
            "You are the Puzzle Solver in the 'Car Puzzle' game.\n"
            "You have been given a car puzzle to solve.\n"
            "There is a specific target car with ID 0 that needs to reach the exit.\n"
            "Provide your solution as a list of moves using the formalized language.\n"
            "Format for moves:\n"
            "[Move] car_id steps\n"
            " - car_id: Integer identifier of the car to move.\n"
            " - steps: Positive or negative integer indicating the number of steps to move (positive for forward, negative for backward).\n"
            "Example: [Move] 0 1\n"

        )
        return prompt

    def get_valid_actions(self, player_id):
        """No restrictions on actions."""
        return None

    def get_info(self):
        """Return additional information."""
        return {}

    def step(self, player_id, action):
        """
        Process the player's action.
        """
        print(action)
        if player_id == 0:
            # puzzle creator
            puzzle_data, solution_steps = self._parse_puzzle_creator_submission(action)
            if not solution_steps:
                if self.render:
                    print(f"Player {player_id} didn't provide solution steps.")
                return None, {player_id:-1, 1-player_id:0}, True, {"reason": f"Player {player_id} didn't provide solution steps."}
            print(puzzle_data)
            print(solution_steps)
            success, info = self._initialize_board(puzzle_data)
            if not success:
                if self.render:
                    print(info["reason"])
                return None, {player_id:-1, 1-player_id:0}, True, info

            # debug, render board
            self.render_board()
            is_valid = self._verify_puzzle_and_solution(solution_steps)
            input(is_valid)

            exit()



            input()
        elif player_id == 1:
            # puzzle solver
            pass 
        input(action)
        # render the game
        #   if self.render:


        #                "\n"
        #    "Here is the puzzle:\n"
        #    f"{self._get_puzzle_description()}\n"
        #    "Provide your solution steps as a numbered list.\n"


        if self.game_over:
            if self.render:
                print("Game is already over.")
            return None, {0: 0, 1: 0}, True, {"reason": "Game has already ended."}

        if self.phase == 1 and player_id == 0:
            # Phase 1: Player 0 provides puzzle and solution
            try:
                puzzle_data, solution_steps = self._parse_puzzle_creator_submission(action)
                self._initialize_board(puzzle_data)
                is_valid = self._verify_puzzle_and_solution(solution_steps)
                if is_valid:
                    if self.render:
                        print("Player 0 provided a valid puzzle and solution.")
                        self.render_board()
                    # Proceed to Phase 2
                    self.phase = 2
                    self.player_prompts[1] = self._generate_puzzle_solver_prompt()
                    info = {"info": "Valid puzzle and solution provided. Player 1, please solve the puzzle."}
                else:
                    if self.render:
                        print("Player 0 provided an invalid puzzle or solution.")
                    # Player 0 loses
                    self.game_over = True
                    reward = {0: -1, 1: 1}
                    info = {"reason": "Invalid puzzle or solution provided by Player 0. Player 1 wins."}
                    return None, reward, True, info
            except Exception as e:
                if self.render:
                    print(f"Error parsing Player 0's submission: {e}")
                # Player 0 loses due to error
                self.game_over = True
                reward = {0: -1, 1: 1}
                info = {"reason": f"Error parsing submission: {e}. Player 1 wins."}
                return None, reward, True, info

            # Successful Phase 1
            return None, {0: 1, 1: 0}, False, info

        elif self.phase == 2 and player_id == 1:
            # Phase 2: Player 1 provides solution
            try:
                solution_steps = self._parse_puzzle_solver_submission(action)
                is_solved = self._apply_solution(solution_steps)
                if is_solved:
                    if self.render:
                        print("Player 1 successfully solved the puzzle.")
                        self.render_board()
                    # Player 1 wins
                    self.game_over = True
                    reward = {0: -1, 1: 1}
                    info = {"reason": "Player 1 solved the puzzle. Player 1 wins."}
                else:
                    if self.render:
                        print("Player 1 failed to solve the puzzle.")
                        self.render_board()
                    # Player 1 loses
                    self.game_over = True
                    reward = {0: 1, 1: -1}
                    info = {"reason": "Player 1 failed to solve the puzzle. Player 0 wins."}
            except Exception as e:
                if self.render:
                    print(f"Error parsing Player 1's submission: {e}")
                # Player 1 loses due to error
                self.game_over = True
                reward = {0: 1, 1: -1}
                info = {"reason": f"Error parsing submission: {e}. Player 0 wins."}
                return None, reward, True, info

            return None, reward, True, info

        else:
            if self.render:
                print(f"Unexpected action from Player {player_id} in phase {self.phase}.")
            return None, {0: 0, 1: 0}, False, {"info": "Unexpected action."}

    def _parse_puzzle_creator_submission(self, submission):
        """
        Parse the submission from the Puzzle Creator (Player 0).
        Expects a specific format with 'Puzzle:' and 'Solution:' sections.
        """
        lines = submission.strip().split('\n')
        puzzle_data = []
        solution_steps = []
        in_solution = False
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if re.match(r'^\[Move\]', line):
                in_solution = True
            if in_solution:
                solution_steps.append(line)
            else:
                puzzle_data.append(line)
        return puzzle_data, solution_steps

    def _parse_puzzle_solver_submission(self, submission):
        """
        Parse the submission from the Puzzle Solver (Player 1).
        Expects a 'Solution:' section.
        """
        lines = submission.strip().split('\n')
        solution_steps = []
        for line in lines:
            line = line.strip()
            if re.match(r'^\[Move\]', line):
                solution_steps.append(line)
        if not solution_steps:
            raise ValueError("No solution steps provided.")
        return solution_steps

    def _initialize_board(self, puzzle_data):
        """
        Initialize the game board based on the puzzle data.
        """
        self.cars = {}
        self.board = None
        self.exit = None
        for line in puzzle_data:
            tokens = line.strip().split()
            if not tokens:
                continue
            if tokens[0] == '[GridSize]':
                if len(tokens) != 3:
                    raise ValueError("Invalid [GridSize] format.")
                width, height = int(tokens[1]), int(tokens[2])
                self.board = [['.' for _ in range(width)] for _ in range(height)]
            elif tokens[0] == '[PlaceCar]':
                if len(tokens) != 6:
                    raise ValueError("Invalid [PlaceCar] format.")
                car_id = int(tokens[1])
                orientation = tokens[2]
                x, y = int(tokens[3]), int(tokens[4])
                length = int(tokens[5])
                success, info = self._place_car(car_id, orientation, x, y, length)
                if not success:
                    return False, info
            elif tokens[0] == '[Exit]':
                if len(tokens) != 3:
                    raise ValueError("Invalid [Exit] format.")
                x, y = int(tokens[1]), int(tokens[2])
                self.exit = (x, y)
        if self.board is None:
            return False, {"reason": "Board size was not specified"}
        if 0 not in self.cars:
            return False, {"reason": "Target car with car_id=0 was not placed"}
        if self.exit is None:
            # Default exit is the right edge on the same row as the target car
            main_car = self.cars.get(0)
            if main_car['orientation'] != 'H':
                return False, {"reason": "Target car must be horizontal for default exit position"}
            self.exit = (len(self.board[0]) - 1, main_car['y'])

        # Ensure all cars are in straight lines and at least two squares long
        for car_id, car in self.cars.items():
            if car['length'] < 2:
                return False, {"reason": f"Car {car_id} must be at least 2 units long."}
            if car['orientation'] not in ('H', 'V'):
                return False, {"reason": f"Car {car_id} has invalid orientation: {car['orientation']}"}

        return True, None

    def _place_car(self, car_id, orientation, x, y, length):
        """
        Place a car on the board.
        """
        if length < 2:
            return False, {"reason": f"Car {car_id} must be at least 2 units long."}
        if orientation not in ('H', 'V'):
            return False, {"reason": f"Invalid orientation for car {car_id}: {orientation}"}
        if car_id in self.cars:
            return False, {"reason": f"Car {car_id} already placed."}
        positions = []
        for i in range(length):
            if orientation == 'H':
                xi, yi = x + i, y
            else:
                xi, yi = x, y + i
            if not (0 <= xi < len(self.board[0]) and 0 <= yi < len(self.board)):
                return False, {"reason": f"Car {car_id} is out of bounds at ({xi}, {yi})."}
            if self.board[yi][xi] != '.':
                return false, {"reason": f"Cell ({xi}, {yi}) is already occupied by car {self.board[yi][xi]}."}
            positions.append((xi, yi))
        for xi, yi in positions:
            self.board[yi][xi] = str(car_id)
        self.cars[car_id] = {
            'orientation': orientation,
            'positions': positions,
            'length': length,
            'x': x,
            'y': y
        }
        return True, None

    def _verify_puzzle_and_solution(self, solution_steps):
        """
        Verify that the puzzle is valid and the solution solves it.
        """
        # Check that all cars are properly placed
        if not self.cars:
            raise ValueError("No cars placed on the board.")
        # Copy the board and cars for simulation
        board_copy = [row.copy() for row in self.board]
        cars_copy = {cid: car.copy() for cid, car in self.cars.items()}
        # Simulate the moves
        for step in solution_steps:
            if not step.startswith('[Move]'):
                raise ValueError(f"Invalid step format: {step}")
            tokens = step.strip().split()
            if len(tokens) != 3:
                raise ValueError(f"Invalid move format: {step}")
            car_id = int(tokens[1])
            steps = int(tokens[2])
            if car_id not in cars_copy:
                raise ValueError(f"Car {car_id} does not exist.")
            success = self._move_car(cars_copy, board_copy, car_id, steps)
            if not success:
                return False
            # Check if target car has reached the exit
            if car_id == 0 and self._check_exit(cars_copy[0]):
                return True
        # After all moves, check if target car has reached the exit
        return self._check_exit(cars_copy[0])

    def _apply_solution(self, solution_steps):
        """
        Apply the solution steps provided by the solver.
        """
        # Copy the board and cars for simulation
        board_copy = [row.copy() for row in self.board]
        cars_copy = {cid: car.copy() for cid, car in self.cars.items()}
        # Simulate the moves
        for step in solution_steps:
            if not step.startswith('[Move]'):
                if self.render:
                    print(f"Invalid step format: {step}")
                return False
            tokens = step.strip().split()
            if len(tokens) != 3:
                if self.render:
                    print(f"Invalid move format: {step}")
                return False
            car_id = int(tokens[1])
            steps = int(tokens[2])
            if car_id not in cars_copy:
                if self.render:
                    print(f"Car {car_id} does not exist.")
                return False
            success = self._move_car(cars_copy, board_copy, car_id, steps)
            if not success:
                if self.render:
                    print(f"Move failed: {step}")
                return False
            # Check if target car has reached the exit
            if car_id == 0 and self._check_exit(cars_copy[0]):
                return True
        # After all moves, check if target car has reached the exit
        return self._check_exit(cars_copy[0])

    def _move_car(self, cars, board, car_id, steps):
        """
        Move a car on the board.
        """
        car = cars[car_id]
        orientation = car['orientation']
        positions = car['positions']
        dx, dy = 0, 0
        if orientation == 'H':
            dx = steps
        else:
            dy = steps
        new_positions = []
        for xi, yi in positions:
            xi_new, yi_new = xi + dx, yi + dy
            if not (0 <= xi_new < len(board[0]) and 0 <= yi_new < len(board)):
                return False  # Move out of bounds
            # Check for collision
            if board[yi_new][xi_new] != '.' and board[yi_new][xi_new] != str(car_id):
                return False  # Collision detected
            new_positions.append((xi_new, yi_new))
        # Update board
        for xi, yi in positions:
            board[yi][xi] = '.'
        for xi, yi in new_positions:
            board[yi][xi] = str(car_id)
        # Update car positions
        car['positions'] = new_positions
        car['x'], car['y'] = new_positions[0]
        return True

    def _check_exit(self, car):
        """
        Check if the target car has reached the exit.
        """
        for xi, yi in car['positions']:
            if (xi, yi) == self.exit:
                return True
        return False

    def _get_puzzle_description(self):
        """
        Generate a description of the puzzle for the solver.
        """
        description = ""
        description += f"[GridSize] {len(self.board[0])} {len(self.board)}\n"
        for car_id, car in self.cars.items():
            orientation = car['orientation']
            x, y = car['x'], car['y']
            length = car['length']
            description += f"[PlaceCar] {car_id} {orientation} {x} {y} {length}\n"
        description += f"[Exit] {self.exit[0]} {self.exit[1]}"
        return description

    def render_board(self):
        """
        Render the current state of the board for visualization.
        """
        print("\nCurrent Board State:")
        for y, row in enumerate(self.board):
            row_display = ''
            for x, cell in enumerate(row):
                if (x, y) == self.exit:
                    row_display += 'E '  # Represent exit with 'E'
                elif cell == '.':
                    row_display += '. '
                elif int(cell) == 0:
                    row_display += 'T '  # Represent target car with 'T'
                else:
                    row_display += f"{cell} "
            print(row_display.strip())
        print(f"Exit at: {self.exit}\n")
