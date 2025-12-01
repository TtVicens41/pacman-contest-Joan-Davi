# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# Import necessary libraries
import random  # For random choices when breaking ties between equally good actions
import contest.util as util  # Utility functions like Counter (a special dictionary)

from contest.capture_agents import CaptureAgent  # Base class for all agents
from contest.game import Directions  # Constants for directions (North, South, East, West, Stop)
from contest.util import nearest_point  # Helper to round positions to nearest grid point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='HybridAgent', second='HybridAgent', num_training=0):
    """
    This function is called at the start of the game to create your team.
    
    Parameters:
    - first_index: The index number for your first agent (e.g., 0 or 2)
    - second_index: The index number for your second agent (e.g., 1 or 3)
    - is_red: True if you're the red team, False if you're the blue team
    - first: Name of the class to use for the first agent (default: 'SamrtOffensiveReflexAgent')
    - second: Name of the class to use for the second agent (default: 'DefensiveReflexAgent')
    - num_training: Number of training games (not used in this implementation)
    
    Returns:
    - A list containing two agent objects that will play the game
    
    Example: If first='SamrtOffensiveReflexAgent', this will create an instance
    of the SamrtOffensiveReflexAgent class with the given index number.
    """
    # eval() converts the string name into the actual class and creates instances
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions.
    
    "Reflex" means the agent makes decisions based on immediate evaluation
    of possible actions, without planning multiple steps ahead.
    
    This is a parent class that other agents will inherit from.
    """

    def __init__(self, index, time_for_computing=.1):
        """
        Initialize the agent.
        
        Parameters:
        - index: The agent's unique ID number in the game
        - time_for_computing: Time limit per turn (in seconds) for computing distances
        """
        # Call the parent class (CaptureAgent) constructor
        super().__init__(index, time_for_computing)
        # Store the starting position (will be set later in register_initial_state)
        self.start = None

    def register_initial_state(self, game_state):
        """
        Called once at the very beginning of the game, before any moves are made.
        Used to store initial information and set up the agent.
        
        Parameters:
        - game_state: The current state of the game (positions, scores, etc.)
        """
        # Remember where we started (useful for returning home at the end)
        self.start = game_state.get_agent_position(self.index)
        # Call parent class initialization (sets up distance calculator, etc.)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        This is called every turn to decide what action the agent should take.
        It picks the action(s) with the highest evaluation score.
        
        Parameters:
        - game_state: The current state of the game
        
        Returns:
        - A direction to move (North, South, East, West, or Stop)
        """
        # Get all legal actions this agent can take (can't move through walls)
        actions = game_state.get_legal_actions(self.index)

        # Evaluate each action to see how good it is
        # This creates a list of scores, one for each action
        values = [self.evaluate(game_state, a) for a in actions]

        # Find the highest score among all actions
        max_value = max(values)
        
        # Keep only the actions that have the maximum score
        # If multiple actions are equally good, they all get included
        # Example: if North=10, South=5, East=10, West=3, best_actions=['North', 'East']
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        # Special case: When there are only 2 or fewer food pellets left
        # Just return to the starting position to end the game
        food_left = len(self.get_food(game_state).as_list())
        if food_left <= 2:
            best_dist = 9999  # Start with a very large distance
            best_action = None
            # Try each action and see which one gets us closest to start
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        # Randomly choose one of the best actions (breaks ties randomly)
        # This adds unpredictability to the agent's behavior
        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Predicts what the game state will look like after taking an action.
        
        Parameters:
        - game_state: The current state of the game
        - action: The action we want to simulate (e.g., 'North')
        
        Returns:
        - The game state that results from taking that action
        
        Note: Sometimes Pacman moves slowly and only covers half a grid square.
        This function handles that edge case.
        """
        # Generate what the state would be after this action
        successor = game_state.generate_successor(self.index, action)
        # Get our position in that future state
        pos = successor.get_agent_state(self.index).get_position()
        
        # Check if we're at a full grid position or halfway between two squares
        if pos != nearest_point(pos):
            # We're only halfway there, so simulate one more step
            return successor.generate_successor(self.index, action)
        else:
            # We're at a full grid position, return the successor
            return successor

    def evaluate(self, game_state, action):
        """
        Calculates a numerical score for how good an action is.
        
        This uses a weighted sum: score = feature1*weight1 + feature2*weight2 + ...
        
        Parameters:
        - game_state: The current state of the game
        - action: The action to evaluate
        
        Returns:
        - A number representing how good this action is (higher = better)
        
        Example: If distance_to_food=5 with weight=-1, and successor_score=10 with weight=100,
        the total evaluation would be: (5 * -1) + (10 * 100) = -5 + 1000 = 995
        """
        # Get the features (measurements) for this action
        features = self.get_features(game_state, action)
        # Get the weights (importance) for each feature
        weights = self.get_weights(game_state, action)
        # Multiply features by weights and sum them up
        # The * operator here does: sum(features[key] * weights[key] for each key)
        return features * weights

    def get_features(self, game_state, action):
        """
        Extracts numerical features (measurements) from a game state + action.
        Features describe important aspects of the situation.
        
        This base version only has one feature, but child classes will add more.
        
        Parameters:
        - game_state: The current state of the game
        - action: The action being considered
        
        Returns:
        - A Counter (dictionary) with feature names as keys and values as numbers
        """
        # Counter is like a dictionary that defaults to 0 for missing keys
        features = util.Counter()
        # Get the state after taking this action
        successor = self.get_successor(game_state, action)
        # The only feature here is the score (positive if winning, negative if losing)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Returns how important each feature is (the weights).
        
        Negative weights mean we want to minimize that feature.
        Positive weights mean we want to maximize that feature.
        
        Parameters:
        - game_state: The current state of the game
        - action: The action being considered
        
        Returns:
        - A dictionary with feature names as keys and importance values as numbers
        """
        # Weight of 1.0 for successor_score means we care about the score
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    An offensive agent that focuses on collecting food from the opponent's side.
    
    This agent inherits from ReflexCaptureAgent and adds food-seeking behavior.
    It tries to minimize distance to food and maximize food collection.
    """

    def get_features(self, game_state, action):
        """
        Extracts features relevant to offense (collecting food).
        
        Features:
        - successor_score: Negative number of food pellets remaining (fewer is better)
        - distance_to_food: Distance to the nearest food pellet
        
        Parameters:
        - game_state: The current state of the game
        - action: The action being evaluated
        
        Returns:
        - A Counter with feature values
        """
        features = util.Counter()
        # Get the state after taking this action
        successor = self.get_successor(game_state, action)
        # Get list of food positions we need to collect
        food_list = self.get_food(successor).as_list()
        
        # Negative length means: fewer food = higher score (we want to eat food)
        features['successor_score'] = -len(food_list)

        # Calculate distance to the nearest food pellet
        if len(food_list) > 0:  # Make sure there's food left
            # Get our position in the successor state
            my_pos = successor.get_agent_state(self.index).get_position()
            # Find the closest food pellet
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        
        return features

    def get_weights(self, game_state, action):
        """
        Returns weights for offensive features.
        
        - successor_score: 100 (strongly favor states with less food remaining)
        - distance_to_food: -1 (prefer being closer to food, so negative weight)
        
        The negative weight on distance means: smaller distance = higher evaluation
        """
        return {'successor_score': 100, 'distance_to_food': -1}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A defensive agent that protects our territory from invaders.
    
    This agent patrols our side of the map and chases enemy Pacmen
    that are trying to eat our food.
    """

    def get_features(self, game_state, action):
        """
        Extracts features relevant to defense.
        
        Features:
        - on_defense: 1 if we're on our side (ghost mode), 0 if on their side (pacman mode)
        - num_invaders: How many enemy Pacmen are on our side
        - invader_distance: Distance to the nearest invader
        - stop: 1 if the action is STOP (we want to discourage standing still)
        - reverse: 1 if the action reverses our current direction (inefficient)
        
        Parameters:
        - game_state: The current state of the game
        - action: The action being evaluated
        
        Returns:
        - A Counter with feature values
        """
        features = util.Counter()
        # Get the state after taking this action
        successor = self.get_successor(game_state, action)

        # Get our state and position in the successor
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Check if we're on defense (on our side = ghost mode)
        # is_pacman is True when on opponent's side, False when on our side
        features['on_defense'] = 1
        if my_state.is_pacman: 
            features['on_defense'] = 0  # We crossed to their side

        # Find enemy invaders (enemies that are Pacmen on our side)
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        # Filter for enemies that are Pacmen and whose position we can see
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        
        # Count how many invaders there are
        features['num_invaders'] = len(invaders)
        
        # If there are invaders, find the distance to the closest one
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        # Penalize stopping (standing still is usually bad)
        if action == Directions.STOP: 
            features['stop'] = 1
        
        # Penalize reversing direction (going back the way we came is inefficient)
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: 
            features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        """
        Returns weights for defensive features.
        
        - num_invaders: -1000 (STRONGLY avoid having invaders; chase them down)
        - on_defense: 100 (prefer staying on our side)
        - invader_distance: -10 (get closer to invaders to catch them)
        - stop: -100 (don't stand still)
        - reverse: -2 (slight penalty for reversing direction)
        
        Note: Large negative weight on num_invaders means catching invaders
        is the top priority.
        """
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}
    
class HybridAgent(ReflexCaptureAgent):

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.offensive_moves_left = 0
        self.time_with_no_enemy = 0
    
    def choose_action(self, game_state):
        action = super().choose_action(game_state)
        
        # Check if we just ate an enemy Pacman
        old_game_state = self.get_previous_observation()
        if old_game_state is None:
            return action
        
        my_pos = game_state.get_agent_position(self.index)
        my_state = game_state.get_agent_state(self.index)
        
        # Only check if we're a ghost (not Pacman) and not scared
        if not my_state.is_pacman and my_state.scared_timer == 0:
            old_enemy_pacman = [old_game_state.get_agent_state(i) for i in self.get_opponents(old_game_state) if old_game_state.get_agent_state(i).is_pacman and old_game_state.get_agent_state(i).get_position() is not None]
            
            # Check each visible Pacman from before
            for old_pacman in old_enemy_pacman:
                old_pos = old_pacman.get_position()
                # Check if we're at that Pacman's old position (we ate it!)
                if my_pos == old_pos:
                    print(f"Agent {self.index} ate an enemy Pacman at {my_pos}!")
                    self.offensive_moves_left += 80
                    break

        return action

    def get_features(self, game_state, action):
        if self.offensive_moves_left > 0:
            self.offensive_moves_left -= 1
            return self.get_offensive_features(game_state, action)
        else:
            return self.get_defensive_features(game_state, action)

    def get_offensive_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        if len(food_list) > 0:
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        
        features['food_left'] = -len(food_list)
        
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None and a.scared_timer == 0]
        if len(ghosts) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in ghosts]
            min_ghost_distance = min(dists)
            features['ghost_distance'] = min_ghost_distance

        if action == Directions.STOP:
            features['stop'] = 1
        
        return features

    
    def get_defensive_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)

        if len(invaders) > 0:
            my_pos = successor.get_agent_state(self.index).get_position()
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            distance_to_invader = min(dists)
            features['invader_distance'] = distance_to_invader
        
        distance_to_border = self.get_distance_to_border(game_state, successor.get_agent_state(self.index).get_position())
        features['distance_to_border'] = distance_to_border

        if action == Directions.STOP: 
            features['stop'] = 1
        
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: 
            features['reverse'] = 1

        my_state = successor.get_agent_state(self.index)
        features['on_defense'] = 1
        if my_state.is_pacman: 
            features['on_defense'] = 0  # We crossed to their side

        allies = [successor.get_agent_state(i) for i in self.get_team(successor) if i != self.index]
        if len(allies) > 0:
            my_pos = successor.get_agent_state(self.index).get_position()
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in allies if a.get_position() is not None]
            if len(dists) > 0:
                features['distance_to_ally'] = min(min(dists), 11)
        
        return features
        
        
    def get_distance_to_border(self, game_state, pos):
        """Calculate distance to the border of our territory"""
        board_width = game_state.data.layout.width
        board_height = game_state.data.layout.height
        
        # Determine which side we're on
        if self.red:
            border_x = board_width // 2 - 4
        else:
            border_x = board_width // 2 + 3
        
        # Get all valid border positions (no walls)
        border_positions = [(border_x, y) for y in range(board_height) 
                           if not game_state.has_wall(border_x, y)]
        
        if len(border_positions) > 0:
            return min([self.get_maze_distance(pos, border_pos) for border_pos in border_positions])
        return 0

    
    def get_weights(self, game_state, action):
        if self.offensive_moves_left > 0:
            return {'distance_to_food': -1, 'ghost_distance': 100, 'food_left': 10, 'stop': -100}
        else:
            return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2, 'distance_to_border': -10, 'distance_to_ally': 10}
