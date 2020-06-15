
# this is a code from watching youtube

import gym
import numpy as np

env = gym.make("MountainCar-v0")


print(env.observation_space.high)
print(env.observation_space.low)
print(env.action_space.n)
print(env.goal_position)

LEARNING_RATE = 0.1;
DISCOUNT =0.95;
EPISODES=2000;

SHOW_EVERY=200;



SampleCounter = 20;

DISCRETE_OS_SIZE = [SampleCounter]*len(env.observation_space.high)

#rint(DISCRETE_OS_SIZE)

discrete_os_win_size = (env.observation_space.high-env.observation_space.low)/DISCRETE_OS_SIZE

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

#rint(q_table.shape)

def get_discrete_state(state):
	discrete_state = (state - env.observation_space.low) / discrete_os_win_size
	return tuple(discrete_state.astype(np.int))



for episode in range(EPISODES):
	if episode % SHOW_EVERY ==0:
		#print(episode)
		render = False
	else: 
		render = False



	discrete_state = get_discrete_state(env.reset())

	done = False
	i = 0;

	while not done:
		i=i+1;
		#print(i)

		action = np.argmax(q_table[discrete_state])
		new_state, reward, done, _ = env.step(action)
		#rint(new_state)
		#print(reward)
		#rint(done)

		new_discrete_state = get_discrete_state(new_state)

		if render:
			env.render()

		if not done:
			max_future_q = np.max(q_table[new_discrete_state])
			current_q = q_table[discrete_state + (action,)]
			new_q = (1-LEARNING_RATE)*current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
			q_table[discrete_state + (action,)] = new_q

		elif new_state[0]>= env.goal_position:
			#reak

			#rint('we made it on episode')
			#rint(episode)
			q_table[discrete_state + (action,)] = 0
			#nv.render()
			#one = True;
			


		discrete_state = new_discrete_state




env.close()


