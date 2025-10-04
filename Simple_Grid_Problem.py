import random

class GridWorld:
    def __init__(self, width, height, start, goal, obstacles=[]):
        # Initialize grid dimensions, start & goal positions, and obstacles
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.reset()

    def reset(self):
        # Reset the agent to the start position
        self.agent_pos = self.start
        return self.agent_pos

    def render(self):
        # Print the current grid with agent, start, goal, and obstacles
        for i in range(self.height):
            row = ''
            for j in range(self.width):
                if (i, j) == self.agent_pos:
                    row += 'A '  # Agent's current position
                elif (i, j) == self.start:
                    row += 'S '  # Start position
                elif (i, j) == self.goal:
                    row += 'G '  # Goal position
                elif (i, j) in self.obstacles:
                    row += 'O '  # Obstacle
                else:
                    row += '. '  # Empty cell
            print(row)
        print()

    def step(self, action):
        # Move the agent in the given direction: 'up', 'down', 'left', 'right'
        x, y = self.agent_pos
        if action == 'up':
            x -= 1
        elif action == 'down':
            x += 1
        elif action == 'left':
            y -= 1
        elif action == 'right':
            y += 1
        else:
            print("Invalid action! Use: up, down, left, right")

        # Check if new position is within grid boundaries
        if 0 <= x < self.height and 0 <= y < self.width:
            new_pos = (x, y)
        else:
            new_pos = self.agent_pos  # Stay in place if moving out of bounds

        self.agent_pos = new_pos

        # Determine reward and whether the episode is done
        if new_pos == self.goal:
            reward = 1  # Reached goal
            done = True
        elif new_pos in self.obstacles:
            reward = -1  # Hit an obstacle
            done = True
        else:
            reward = -0.01  # Small penalty for each step
            done = False

        return new_pos, reward, done


# -----------------------------
# User-controlled agent with RANDOM obstacles
# -----------------------------
width, height = 5, 5
start = (0, 0)
goal = (4, 4)

# Generate random obstacles (3 positions each run)
all_positions = [(i, j) for i in range(height) for j in range(width)]
all_positions.remove(start)
all_positions.remove(goal)
obstacles = random.sample(all_positions, 3)

env = GridWorld(width, height, start, goal, obstacles)
state = env.reset()
done = False

print("Welcome to Grid World!\n")
print("Game Rules:")
print(" - Reach the Goal (G) to win 🎉")
print(" - Avoid Obstacles (O), or you lose 💀")
print(" - You are the Agent (A), starting from Start (S)")
print(" - Use commands: up, down, left, right to move\n")

env.render()

while not done:
    action = input("Enter your move (up, down, left, right): ")
    state, reward, done = env.step(action)
    print(f"New State: {state}, Reward: {reward}")
    env.render()

if state == goal:
    print("🎉 Congratulations! You reached the goal!")
else:
    print("💀 Game Over! You hit an obstacle.")



/*
---------------------------------- Output 1 --------------------------------

Welcome to Grid World!

Game Rules:
 - Reach the Goal (G) to win 🎉
 - Avoid Obstacles (O), or you lose 💀
 - You are the Agent (A), starting from Start (S)
 - Use commands: up, down, left, right to move

A . . . O 
O . . . . 
. . . O . 
. . . . . 
. . . . G 

Enter your move (up, down, left, right): right
New State: (0, 1), Reward: -0.01
S A . . O 
O . . . . 
. . . O . 
. . . . . 
. . . . G 

Enter your move (up, down, left, right): down
New State: (1, 1), Reward: -0.01
S . . . O 
O A . . . 
. . . O . 
. . . . . 
. . . . G 

Enter your move (up, down, left, right): down
New State: (2, 1), Reward: -0.01
S . . . O 
O . . . . 
. A . O . 
. . . . . 
. . . . G 

Enter your move (up, down, left, right): down
New State: (3, 1), Reward: -0.01
S . . . O 
O . . . . 
. . . O . 
. A . . . 
. . . . G 

Enter your move (up, down, left, right): right
New State: (3, 2), Reward: -0.01
S . . . O 
O . . . . 
. . . O . 
. . A . . 
. . . . G 

Enter your move (up, down, left, right): right
New State: (3, 3), Reward: -0.01
S . . . O 
O . . . . 
. . . O . 
. . . A . 
. . . . G 

Enter your move (up, down, left, right): right
New State: (3, 4), Reward: -0.01
S . . . O 
O . . . . 
. . . O . 
. . . . A 
. . . . G 

Enter your move (up, down, left, right): down
New State: (4, 4), Reward: 1
S . . . O 
O . . . . 
. . . O . 
. . . . . 
. . . . A 

🎉 Congratulations! You reached the goal!


---------------------------------- Output 2 --------------------------------

Welcome to Grid World!

Game Rules:
 - Reach the Goal (G) to win 🎉
 - Avoid Obstacles (O), or you lose 💀
 - You are the Agent (A), starting from Start (S)
 - Use commands: up, down, left, right to move

A . . . . 
. . . O . 
. . O . . 
. . . O . 
. . . . G 

Enter your move (up, down, left, right): down
New State: (1, 0), Reward: -0.01
S . . . . 
A . . O . 
. . O . . 
. . . O . 
. . . . G 

Enter your move (up, down, left, right): down
New State: (2, 0), Reward: -0.01
S . . . . 
. . . O . 
A . O . . 
. . . O . 
. . . . G 

Enter your move (up, down, left, right): right
New State: (2, 1), Reward: -0.01
S . . . . 
. . . O . 
. A O . . 
. . . O . 
. . . . G 

Enter your move (up, down, left, right): right
New State: (2, 2), Reward: -1
S . . . . 
. . . O . 
. . A . . 
. . . O . 
. . . . G 

💀 Game Over! You hit an obstacle.


*/