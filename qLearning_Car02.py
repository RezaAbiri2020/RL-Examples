
# RL for mountain car 

import gym
import numpy as np

env = gym.make("MountainCar-v0")


print(env.observation_space.high)
print(env.observation_space.low)
print(env.action_space.n)
print(env.goal_position)

LEARNING_RATE = 0.1;
DISCOUNT =0.95;
EPISODES=25000;

SHOW_EVERY=1000;



SampleCounter = 20;

DISCRETE_OS_SIZE = [SampleCounter]*len(env.observation_space.high)

#rint(DISCRETE_OS_SIZE)

discrete_os_win_size = (env.observation_space.high-env.observation_space.low)/DISCRETE_OS_SIZE



epsilon = 0.5;
sTART_EPLISON_DECAYING = 1;
END_EPLISON_DECAYING  = EPISODES // 2;
epsilon_decay_value = epsilon / (END_EPLISON_DECAYING - sTART_EPLISON_DECAYING); 


q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

#rint(q_table.shape)

def get_discrete_state(state):
	discrete_state = (state - env.observation_space.low) / discrete_os_win_size
	return tuple(discrete_state.astype(np.int))



for episode in range(EPISODES):
	if episode % SHOW_EVERY ==0:
		print(episode)
		render = True 
	else: 
		render = False



	discrete_state = get_discrete_state(env.reset())

	done = False
	i = 0;

	while not done:
		i=i+1;
		#print(i)
		if np.random.random() > epsilon:
			action = np.argmax(q_table[discrete_state])
		else: 
			action = np.random.randint(0, env.action_space.n)

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

			print('we made it on episode')
			print(episode)
			q_table[discrete_state + (action,)] = 0
			#nv.render()
			#one = True;
			


		discrete_state = new_discrete_state
	if END_EPLISON_DECAYING >=  episode >= sTART_EPLISON_DECAYING: 
		epsilon -=epsilon_decay_value




env.close()


