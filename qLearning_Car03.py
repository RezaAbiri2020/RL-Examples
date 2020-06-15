
# RL for mountain car 

import gym
import numpy as np
import time 

env = gym.make("MountainCar-v0")


print(env.observation_space.high)
print(env.observation_space.low)
print(env.action_space.n)
print(env.goal_position)


LEARNING_RATE = 1;
DISCOUNT =0.95;
EPISODES=50;

SHOW_EVERY=10;



SampleCounter_SizeQ = 40;

DISCRETE_OS_SIZE = [SampleCounter_SizeQ]*len(env.observation_space.high)

#rint(DISCRETE_OS_SIZE)

discrete_os_win_size = (env.observation_space.high-env.observation_space.low)/DISCRETE_OS_SIZE

q_table = np.random.uniform(low=0	, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

print(q_table)

#rint(q_table.shape)

def get_discrete_state(state):
	discrete_state = (state - env.observation_space.low) / discrete_os_win_size
	return tuple(discrete_state.astype(np.int))



for episode in range(EPISODES):
	if episode % SHOW_EVERY ==0:
		#print(episode)
		render = True
	else: 
		render = False



	discrete_state = get_discrete_state(env.reset())

	done = False

	i = 0;

	IterationCounter = 1601;

	while i < IterationCounter:
		i=i+1;
		#rint(i)
		if i % 5000==0:
			print(i)

		action = np.argmax(q_table[discrete_state])
		new_state, reward, done, _ = env.step(action)
		#rint(new_state)
		#rint(reward)
		#rint(done)

		new_discrete_state = get_discrete_state(new_state)

		if render:
			env.render()

		if i < IterationCounter:
			max_future_q = np.max(q_table[new_discrete_state])
			current_q = q_table[discrete_state + (action,)]
			new_q = (1-LEARNING_RATE)*current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
			q_table[discrete_state + (action,)] = new_q

		elif new_state[0]>= env.goal_position:
			#reak

			print('we made it on episode')
			print(episode)

			#rint('we made it on iteration:')
			#rint(i)

			q_table[discrete_state + (action,)] = 0
			#nv.render()
			
			#one = True;
			


		discrete_state = new_discrete_state


#ime.sleep(10)



env.close()


