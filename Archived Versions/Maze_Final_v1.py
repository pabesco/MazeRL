import numpy as np
import cv2
import copy
import time
from math import floor
import pickle
from matplotlib import pyplot as plt
from ImageColorInvert import ImageColorInvert


DEFAULT = input('Use Default Settings (y/n)?     ')
if DEFAULT == 'y':
    filename = 'NewMaze2.jpg'
    img_orig = cv2.imread(ImageColorInvert(filename), cv2.IMREAD_COLOR)
else:
    Type = input('Does the input image has black background with white walls? y/n    ')
    file = input('Enter file name: ')
    if Type == 'y':
        img_orig = cv2.imread(file, cv2.IMREAD_COLOR)
        img_orig = cv2.resize(img_orig, (round(img_orig.shape[0]/100)*100, round(img_orig.shape[1]/100)*100))
    else:
        img_orig = cv2.imread(ImageColorInvert(file), cv2.IMREAD_COLOR)

WIDTH, HEIGHT = img_orig.shape[1], img_orig.shape[0]
print(WIDTH, HEIGHT)
cv2.imshow('Image', img_orig)
cv2.waitKey(10)

if DEFAULT == 'y':
    Y_GRIDS = 10
    Starting_Grid_X = 4
    Starting_Grid_Y = 0
    Ending_Grid_X = 5
    Ending_Grid_Y = 9
    UNIT_SIZE = int(HEIGHT/int(Y_GRIDS))
    X_GRIDS = int(WIDTH/UNIT_SIZE)
else:
    Y_GRIDS = input('How many grids along y-axis?')
    UNIT_SIZE = int(HEIGHT/int(Y_GRIDS))
    X_GRIDS = int(WIDTH/UNIT_SIZE)
    plt.imshow(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB))
    plt.xticks(range(round(CENTRE), WIDTH, UNIT_SIZE), list(range(int(X_GRIDS))))
    plt.yticks(range(round(CENTRE), HEIGHT, UNIT_SIZE), list(range(int(Y_GRIDS))))
    #plt.yticks(range(int(Y_GRIDS)))
    plt.show(block=False)
    Starting_Grid_X = int(input('Starting Grid number Horizontal: '))
    Starting_Grid_Y = int(input('Starting Grid number Vertical: '))
    Ending_Grid_X = int(input('Ending Grid number Horizontal: '))
    Ending_Grid_Y = int(input('Ending Grid number Vertical: '))
    plt.close()

CENTRE = UNIT_SIZE/2


STEPS = 100
RED = (0, 0, 255)
GREEN = (0, 255, 0)

HM_EPISODES = 2000

MOVE_PENALTY = 1
#previously both BORDER_PENALTY and WIN_REWARD were 300
#This the agent to jump over a border near the end point due to less number of steps required
#Hence, increased BORDER_PENALTY and reduced WIN_REWARD

BORDER_PENALTY = 1000
WIN_REWARD = 25

epsilon = 0.5
EPS_DECAY = 0.994
SHOW_EVERY = 50

start_q_table = None

LEARNING_RATE = 0.1
DISCOUNT = 0.95


class Blob:
    def __init__(self):
        #print("init block")
        self.x = floor(Starting_Grid_X*UNIT_SIZE + CENTRE)
        self.y = floor(Starting_Grid_Y*UNIT_SIZE + CENTRE)
        self.done = False
        self.step = 0
        self.COLOR = RED

    def __str__(self):
        return f"{self.x}, {self.y}"

    def action(self, choice):
        #print("Action Block")
        self.step += 1
        if choice == 0:
            self.move(choice, x=2, y=0)
        elif choice == 1:
            self.move(choice, x=-2, y=0)
        elif choice == 2:
            self.move(choice, x=0, y=2)
        elif choice == 3:
            self.move(choice, x=0, y=-2)

        #self.bordercross(choice)

    def move(self, choice, x, y):
        #print("Move Block")
        self.xnew = floor(self.x + x*CENTRE)
        self.ynew = floor(self.y + y*CENTRE)
        self.terminal(choice)

        self.x = self.xnew
        self.y = self.ynew

    def terminal(self, choice):
        #print("Border Cross Block")
        if self.xnew < 0 or self.xnew > WIDTH or self.ynew < 0 or self.ynew > HEIGHT:
            self.done = True
        elif choice == 0:
            for i in range(self.x, self.xnew):
                compare_border = img_orig[self.y][i] > (220, 220, 220)
                if compare_border.all():
                    self.done = True
                    break
        elif choice == 1:
            for i in range(self.xnew, self.x):
                compare_border = img_orig[self.y][i] > (220, 220, 220)
                if compare_border.all():
                    self.done = True
                    break
        elif choice == 2:
            for i in range(self.y, self.ynew):
                compare_border = img_orig[i][self.x] > (220, 220, 220)
                if compare_border.all():
                    self.done = True
                    break
        elif choice == 3:
            for i in range(self.ynew, self.y):
                compare_border = img_orig[i][self.x] > (200, 220, 220)
                if compare_border.all():
                    self.done = True
                    break
        if self.done:
            self.reward = -BORDER_PENALTY
        elif self.step > STEPS:
            self.done = True
            self.reward = -MOVE_PENALTY
        elif Ending_Grid_X*UNIT_SIZE < player.xnew < (Ending_Grid_X+1)*UNIT_SIZE \
            and Ending_Grid_Y*UNIT_SIZE <  player.ynew < (Ending_Grid_Y+1)*UNIT_SIZE:
            self.reward = WIN_REWARD
            self.done = True
            print("Hoooorayyyy!!!!")
            self.COLOR = GREEN
            with open(f"qtable-{filename}.pickle", "wb") as f:
                pickle.dump(q_table, f)
        else:
            self.reward = -MOVE_PENALTY

def whichbox(x):
    return int(x/UNIT_SIZE)

start_q_table = f'qtable-{filename}.pickle'
q_table = {}

if start_q_table is None:
    for i in range(int(HEIGHT/UNIT_SIZE)):
        for ii in range(int(WIDTH/UNIT_SIZE)):
            q_table[(i,ii)] = [np.random.uniform(-5,0) for j in range(4)]
else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)
print(len(q_table))
episode_rewards = []

for episode in range(HM_EPISODES):
    player = Blob()
    img = copy.deepcopy(img_orig)

    #print("New Episode")
    episode_reward = 0
    print(episode)
    done = False
    if episode % SHOW_EVERY == 0:
        cv2.circle(img,(player.x, player.y), 15, player.COLOR, -1)
        cv2.imshow('image',img)
        cv2.waitKey(100)

    while not done:
        #print(player)
        img = copy.deepcopy(img_orig)
        obs = (whichbox(player.y), whichbox(player.x))
        #print(obs)

        if np.random.random() < epsilon:
            action = np.random.randint(0,4)
        else:
            action = np.argmax(q_table[obs])
        player.action(action)

        done = player.done
        reward = player.reward

        new_obs = (whichbox(player.y), whichbox(player.x))
        if player.reward == -BORDER_PENALTY:
            max_future_q = -BORDER_PENALTY
        elif player.reward == WIN_REWARD:
            max_future_q = WIN_REWARD
        else:
            max_future_q = np.max(q_table[new_obs])

        current_q = q_table[obs][action]

        if reward == WIN_REWARD:
            new_q = WIN_REWARD
        else:
            new_q = (1-LEARNING_RATE)*current_q + LEARNING_RATE*(reward + DISCOUNT*max_future_q)

        q_table[obs][action] = new_q

        if episode % SHOW_EVERY == 0:
            cv2.circle(img,(player.x, player.y), 15, player.COLOR, -1)
            cv2.imshow('image',img)
            cv2.waitKey(100)

        episode_reward += reward

    episode_rewards.append(episode_reward)

    epsilon *= EPS_DECAY

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY, ))/SHOW_EVERY, mode = 'valid')

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.show()

with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)
