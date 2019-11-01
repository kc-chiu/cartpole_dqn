# Deep Reinforcement Learning for CartPole Game in OpenAI Gym

Week 9 homework of [Make Money with Machine Learning course](https://www.machinelearningcourse.io/courses/make-money) by [Siraj Raval](https://www.youtube.com/channel/UCWN3xxRkmTPmbKwht9FuE5A)

------------------

## Requirements

- Python 3.6+
- Python Modules listed in [requirements.txt](https://github.com/kc-chiu/cartpole_dqn/blob/master/requirements.txt)
- Note that pyglet needs version 1.2.4 for OpenAI Gym to render its games properly

------------------

## Using the program

- All codes are in [main.py](https://github.com/kc-chiu/cartpole_dqn/blob/master/main.py)
- Install Python Modules listed in [requirements.txt](https://github.com/kc-chiu/cartpole_dqn/blob/master/requirements.txt)
- In line 72 of [main.py](https://github.com/kc-chiu/cartpole_dqn/blob/master/main.py):
  - set ```pre_trained``` to ```False``` to train the agent
  - set ```pre_trained``` to ```True``` to use the pre-trained model stored in [model](https://github.com/kc-chiu/cartpole_dqn/tree/master/model) folder
- Run [main.py](https://github.com/kc-chiu/cartpole_dqn/blob/master/main.py)
  - The game is not rendered during training
  - The game will be rendered when using pre-trained model
  - The model 'cartpole-dqn-500.h5' in [model](https://github.com/kc-chiu/cartpole_dqn/tree/master/model) folder can achieve score of 499 in about 80% of the games
- You may try to raise the ```TIME_GOAL``` in line 80 and train the agent to improve the score 
- Change the value of ```EPISODES``` in line 11 for the number of game episodes you need

------------------

## References
1. [deep-q-learning](https://github.com/keon/deep-q-learning)
2. [OpenAI Gym](https://gym.openai.com/docs/)
