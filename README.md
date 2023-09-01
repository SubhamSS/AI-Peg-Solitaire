# AI-Peg-Solitaire
An AI to solve and assist with the Peg Solitaire game using Reinforcement Learning

# Description

[Peg solitaire](https://en.wikipedia.org/wiki/Peg_solitaire) is a classic single-player board game, played on a board
with holes filled with pegs, where the objective is to remove all but one peg by jumping them over each
other. This project aims to achieve the following:
* Implement a [Deep Q-Network](https://en.wikipedia.org/wiki/Q-learning) to solve different board configurations.
* Implement a methodology to assist a human playing the game. This would work as the following: The bot and the human take turns; the bot moves first followed by the human. The bot’s aim is to make the optimal move considering the human’s possible moves.

The code is written in Python and uses [PyTorch](https://pytorch.org/)

## Environment details

The below environment details a 4 x 4 board. As we change the board, 
* State: An array of 4 x 4 represents the state.
  * `0`: No peg present in the position
  * `1`: Peg is present in the position
* Action: A tuple of size 2 representing the start and end positions
  * Action ((4,2),(2,2)) represents peg from position (4,2) to (2,2) over (3,2)
  * Action space consists of all possible actions (e.g.: 40 for a 4x4 board), but only a few valid at each state
* Reward: Reward is defined as:
  * 100 if the minimum number of pegs is achieved
  * Else, Reward = −2^(no of pegs on board)

## Agent details

* The neural network approximates the afterstate value function
  * It has 33 inputs and 1 output
  
## Training loop details  

* episodes loop:
  * reset env
  * env steps loop:
    * evaluate valid actions for the current state
      * get the valid actions from the env (not standard Gym behaviour)
      * figure out the next board state for each valid action
      * push all the next board states through the neural network to predict the afterstate values
    * use the policy to choose an action
      * with epsilon prob: random choice
      * with (1 - epsilon) prob: greedy choice i.e. choose the action with the highest afterstate value estimate
    * step the env with the chosen action
    * calculate the target afterstate value for the current state based on the reward from the env
    added to the discounted estimate of the value for the afterstate resulting from taking the action
    * calculate the loss i.e. the difference between the current estimate of the afterstate
    value of the current state and the calculated target value
    * back propagate the loss  

# Training plots

![Training plots](4_4_train.jpg)

# Setting up

I'm pretty new to Python and Conda etc. but I think the following should do it:

```
conda env create -f environment.yml
conda develop ../gym-solitaire
```

This assumes that you have cloned [gym-solitaire](https://github.com/taylorjg/gym-solitaire) into `../gym-solitaire`. 

# Play

The following command will load a previously trained model and play a single episode of Solitaire:

```
python td_solitaire.py --play
actions: [44, 69, 65, 28, 57, 65, 73, 41, 74, 8, 17, 27, 0, 32, 66, 47, 33, 71, 58, 4, 0, 56, 30, 15, 49, 11, 20, 54, 24, 13, 7]
  ...
  ...
.......
...X...
.......
  ...
  ...
```

# Train

The following command will train a model and, if successful, save the trained model to `td_solitaire.pt`:

```
python td_solitaire.py
```

# Links

* [Peg solitaire](https://en.wikipedia.org/wiki/Peg_solitaire)
* I created a custom [OpenAI Gym](https://gym.openai.com/) environment: 
  * [gym-solitaire](https://github.com/taylorjg/gym-solitaire)
  * [How to create new environments for Gym](https://github.com/openai/gym/blob/master/docs/creating-environments.md)  
* I have done a lot of reading about reinforcement learning but I found the following to be particularly helpful:
  * [_Reinforcement Learning: An Introduction_](http://incompleteideas.net/book/the-book.html) by Richard S. Sutton
and Andrew G. Barto
    * Chapter 6 _Temporal-Difference Learning_
      * Especially Section 6.8 _Games, Afterstates, and Other Special Cases_
    * Section 16.1 _TD-Gammon_
    * [Full Pdf](http://incompleteideas.net/book/RLbook2020.pdf)
  * [Reinforcement Learning in the Game of Othello:
Learning Against a Fixed Opponent
and Learning from Self-Play](https://www.ai.rug.nl/~mwiering/GROUP/ARTICLES/paper-othello.pdf)
