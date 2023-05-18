import jax
import jax.numpy as jnp
import jax.random as random
from loaders.sequence_generator import SequenceGenerator
from utils.utils import load_pickle, make_gif, filters_to_grids
from models.nem import NEMUpdateRule
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

def inner_test_episode(key, update_rule, meta, x, y, x_test, y_test):

    key, subkey = random.split(key, 2)
    base = update_rule.create_base(key, 3, 256, 128, 10, 5)

    def inner_step(acc, input_data):
        x = input_data['x']
        y = input_data['y']
        new_acc = update_rule.update(meta, acc, x, y)
        filters = new_acc['w'][0]
        return new_acc, filters

    #inner train
    input_data = {'x':x, 'y':y}
    base, filters = jax.lax.scan(inner_step, base, input_data)

    #memorization eval
    y_, _, _ = jax.vmap(update_rule.base_forward, in_axes = [None, None, 0])(meta, base, x)
    y_ = jnp.argmax(y_, -1)
    scores = (y_ == y)
    score = scores.mean(0)

    #generalization eval
    y_, _, _ = jax.vmap(update_rule.base_forward, in_axes = [None, None, 0])(meta, base, x_test)
    y_ = jnp.argmax(y_, -1)
    gen_scores = (y_ == y_test)
    gen_score = gen_scores.mean(0)

    return filters, score, gen_score

def get_remember_characteristic(key, update_rule, meta, x, y, x_test, y_test, n_repeat=20):

    scores = 0
    for _ in range(n_repeat):
        key, subkey = random.split(key, 2)
        base = update_rule.create_base(subkey, 3, 256, 128, 10, 5)

        def inner_step(acc, input_data):
            x = input_data['x']
            y = input_data['y']
            new_acc = update_rule.update(meta, acc, x, y)
            filters = new_acc['w'][0]
            return new_acc, filters

        #inner train
        input_data = {'x':x, 'y':y}
        base, filters = jax.lax.scan(inner_step, base, input_data)

        #memorization eval
        y_, _, _ = jax.vmap(update_rule.base_forward, in_axes = [None, None, 0])(meta, base, x)
        y_ = jnp.argmax(y_, -1)
        scores = scores + (y_ == y)
    scores = scores / n_repeat
    return scores

cpu = jax.devices("cpu")[0]
print(cpu)

update_rule = NEMUpdateRule()

data = SequenceGenerator()

key = random.PRNGKey(0)

datasets = ['cifar10', 'mnist', 'svhn']
seq_len = 1000

key, subkey = random.split(key, 2)

#load population and select best individual
meta = jax.device_put(load_pickle('meta_gen_5000.pt'), cpu)
meta = jax.tree_map(lambda x: x[500], meta)
#meta = init_nem_architecture(key, 5, 5, 10)

x, y = data.gen_sequence(dataset_list=datasets, seq_len=5000, correlation='ci', fold='train')
x, y = x.reshape(x.shape[0], -1), y.astype(int)
x_test, y_test = data.gen_sequence(dataset_list=datasets, seq_len=5000, correlation='ci', fold='test')
x_test, y_test = x_test.reshape(x.shape[0], -1), y_test.astype(int)

key, subkey = random.split(key, 2)
filters, score, gen_score = jax.jit(inner_test_episode, static_argnums=[1])(subkey, update_rule, meta, x, y, x_test, y_test)
remember_characteristic = jax.jit(get_remember_characteristic, static_argnums=[1])(subkey, update_rule, meta, x, y, x_test, y_test)

plt.plot(remember_characteristic)
plt.show()

#frames = filters_to_one_filter_grid(filters)
frames = filters_to_grids(filters)
make_gif('test.gif', np.array(frames))
print('score:', score, 'gen_score', gen_score)

