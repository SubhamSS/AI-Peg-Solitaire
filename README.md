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

The below environment details a 4 x 4 board. As we change the board to higher dimensions, the State and Actions remain similar (with only changing in dimensions), while Rewards are modeled differently, which will be discussed later. 
* State: An array of 4 x 4 represents the state.
  * `0`: No peg present in the position
  * `1`: Peg is present in the position
* Action: A tuple of size 2 representing the start and end positions
  * Action ((4,2),(2,2)) represents peg from position (4,2) to (2,2) over (3,2)
  * Action space consists of all possible actions (e.g.: 40 for a 4x4 board), but only a few valid at each state
* Reward: Reward is defined as:
  * 100 if the minimum number of pegs is achieved
  * Else, Reward = <math>−2<sup>(no of pegs on board)</sup></math>

## DQN Algorithm

Peg solitaire's discrete actions suit a DQN framework

<b>Objective of DQN</b>: To learn an optimal policy that maximizes the expected discounted sum of
rewards

While running
* 𝑎 &#8592 argmax <math>𝑄(𝑠,𝑎)</math>
* Add <math>s,a,r</math>,<math>s<sup>'</sup></math> to memory, where <math>s<sup>'</sup></math> = <math>s+a</math>
* If len (memory) > batch_size
  * Sample batch of <math>𝑠, 𝑎, 𝑟, s<sup>'</sup></math>
  * 𝑄𝑡𝑎𝑟𝑔𝑒𝑡← 𝑟+ 𝛾 𝑄′ 𝑠
• 𝑄𝑒𝑥𝑝𝑒𝑐𝑡𝑒𝑑← 𝑄 𝑠
• ℒ 𝜃))← 𝑄𝑡𝑎𝑟𝑔𝑒𝑡−𝑄𝑒𝑥𝑝𝑒𝑐𝑡𝑒𝑑
• 𝑄’’← weights closer to 𝑄
  

# Training plots

<img src="Git images/4_4_train.jpg" width="900">

# Higher Dimensional Boards

As we increase the board dimensions, the number of possible board configurations increases exponentially with the number of pegs, making it challenging to explore and learn from all possible states

Further, the reward for a particular move is not immediately
evident ,and the agent may have to make a series of moves to reach a desirable state.
This can make it difficult for the agent to learn an optimal policy, as it needs to consider
the long term consequences of its actions.

Thus, we look for ways to improve the reward model, and modified the rewards to the following:
<img style="float: right;" src="Git images/4_4_train.jpg">
* 5 x 5  Board: Added extra Reward to states which have valid actions:
  * 10<sup>8</sup>if the minimum number of pegs is achieved
  * Else:
    * If state has valid actions: 2 x 2<sup>16−number of pegs on board</sup>
    * Else : 2<sup>16−number of pegs on board</sup>
* Classical Board: New reward term =
Modified Reward + <math>n x \sum_{i=1}^n d_i </math>
n: the number of
empty holes in the board
•
d: the distance
of the hole from the board’s center

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
