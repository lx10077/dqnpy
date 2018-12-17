import os
import numpy as np
import time
from memory import Memory, ResultsBuffer


def train(env,
          model,
          base_path,
          batch_size=32,
          epsilon=0.01,
          update_every=4,
          update_target_every=1000,
          learning_starts=200,
          memory_size=500000,
          num_iterations=6250000):
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    model_path = os.path.join(base_path, "model")
    begin_i = model.load(model_path)

    memory_buffer = Memory(memory_size)
    results_buffer = ResultsBuffer(base_path)

    state = env.reset()
    for i in range(learning_starts):
        action = np.random.randint(env.action_n) if np.random.uniform() < epsilon else model.get_action(state)
        next_state, reward, done, info = env.step(action)
        memory_buffer.append((state, action, reward, next_state, done))
        state = next_state

    state = env.reset()
    start = time.time()
    for i in range(begin_i+1, num_iterations):
        action = np.random.randint(env.action_n) if np.random.uniform() < epsilon else model.get_action(state)
        next_state, reward, done, info = env.step(action)
        results_buffer.update_info(info)
        memory_buffer.append((state, action, reward, next_state, done))
        state = next_state

        if i > 0 and i % update_every == 0:
            summaries = model.update(*memory_buffer.sample(batch_size))
            results_buffer.update_info(summaries)

        if i > 0 and i % (update_every * update_target_every) == 0:
            model.update_target()
            model.save(model_path, i)

            t = time.time() - start
            print("Save model, global step:{}, delta_time:{}.".format(i, t))
            start = time.time()
