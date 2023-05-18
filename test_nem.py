import jax
import jax.nn as nn
import jax.numpy as jnp
import jax.random as random
from loaders import SequenceGenerator
from random import randint
from utils import save_pickle, load_pickle, make_gif, filters_to_grids, filters_to_one_filter_grid
from model import init_mlp_architecture, init_nem_architecture, apply_forward_nem, apply_meta_net, create_base, update
from genetic import compute_novelty, half_clone_mutate
import numpy as np

def inner_test_episode(key, meta, x, y, x_test, y_test):

    key, subkey = random.split(key, 2)
    base = create_base(key, 3, 256, 128, 10, 5)

    def inner_step(acc, input_data):
        x = input_data['x']
        y = input_data['y']
        new_acc = update(meta, acc, x, y)
        filters = new_acc['w'][0]
        return new_acc, filters

    #inner train
    input_data = {'x':x, 'y':y}
    base, filters = jax.lax.scan(inner_step, base, input_data)

    #memorization eval
    y_, _, _ = jax.vmap(apply_forward_nem, in_axes = [None, None, 0])(meta, base, x)
    y_ = jnp.argmax(y_, -1)
    scores = (y_ == y)
    score = scores.mean(0)

    #generalization eval
    y_, _, _ = jax.vmap(apply_forward_nem, in_axes = [None, None, 0])(meta, base, x_test)
    y_ = jnp.argmax(y_, -1)
    gen_scores = (y_ == y_test)
    gen_score = gen_scores.mean(0)

    return filters, score, gen_score


data = SequenceGenerator()

key = random.PRNGKey(0)
datasets = ['cifar10', 'mnist', 'svhn']
seq_len = 1000

key, subkey = random.split(key, 2)

#load population and select first individual
meta = load_pickle('meta.pt')
meta = jax.tree_map(lambda x: x[500], meta)

x, y = data.gen_sequence(dataset_list=datasets, seq_len=1000, correlation='ci')
x, y = x.reshape(x.shape[0], -1), y.astype(int)
x_test, y_test = data.gen_sequence(dataset_list=datasets, seq_len=1000, correlation='ci')
x_test, y_test = x_test.reshape(x.shape[0], -1), y_test.astype(int)

key, subkey = random.split(key, 2)
filters, score, gen_score = jax.jit(inner_test_episode)(subkey, meta, x, y, x_test, y_test)

frames = filters_to_one_filter_grid(filters)
make_gif('test.gif', np.array(256 * frames).astype(np.uint8))

print('score:', score, 'gen_score', gen_score)


