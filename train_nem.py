import jax
import jax.numpy as jnp
import jax.random as random
from loaders.sequence_generator import SequenceGenerator
from random import randint
from utils.utils import save_pickle, load_pickle
from models.synapse_ur import SynapseUpdateRule
from models.nem import NEMUpdateRule
from models.nem_rfa import NEMRFAUpdateRule
from models.nem_dfa import NEMDFAUpdateRule
from genetic.genetic import compute_novelty, half_clone_mutate, gaussian_mutation, nonlocal_mutation

def get_remember_test_sequence(x, y):
    new_x = jnp.zeros(x.shape)
    new_y = jnp.zeros(y.shape).astype(int)
    for i in range(x.shape[0]):
        r = randint(0, i)
        new_x = new_x.at[i].set(x[r])
        new_y = new_y.at[i].set(y[r])
    return new_x, new_y

def inner_episode(key, update_rule, meta, x, y, x_test, y_test):
    vmap_create_base = jax.vmap(update_rule.create_base, in_axes=[0, None, None, None, None, None])
    vmap_update = jax.vmap(update_rule.update, in_axes=[0, 0, None, None])
    vmap_inference = jax.vmap(update_rule.base_forward, in_axes=[0, 0, None])

    key, subkey = random.split(key, 2)
    keys = random.split(subkey, 1000)
    base = vmap_create_base(keys, 3, 256, 128, 10, 5)

    def inner_step(acc, input_data):
        x = input_data['x']
        y = input_data['y']
        x_test = input_data['x_test']
        y_test = input_data['y_test']
        new_acc = vmap_update(meta, acc, x, y)
        y_test_ = jnp.argmax(vmap_inference(meta, new_acc, x_test)[0], -1)
        e = (y_test_ == y_test)

        return new_acc, e

    input_data = {'x':x, 'y':y, 'x_test':x_test, 'y_test':y_test}
    base, scores = jax.lax.scan(inner_step, base, input_data)
    scores = scores.mean(0)
    diversity = compute_novelty(meta)
    return scores, diversity


update_rule = NEMUpdateRule()

vmap_create_meta = jax.vmap(update_rule.create_meta, in_axes=[0, None, None, None])

data = SequenceGenerator()

key = random.PRNGKey(0)


key, subkey = random.split(key, 2)
keys = random.split(subkey, 1000)
meta = vmap_create_meta(keys, 5, 5, 10)
meta = load_pickle('meta_gen_5000.pt')
logs = {'mean':[], 'max':[], 'diversity':[]}
k = 0
n_repeat = 1
curriculum = [
        #(10,   ['cifar10']),
        #(30,   ['cifar10']),
        #(100,  ['cifar10']),
        #(1000, ['cifar10', 'mnist', 'svhn']),
        (5000, ['cifar10', 'mnist', 'svhn']),
]

for cur_seq_len, datasets in curriculum:
    while True:
        scores = 0
        for _ in range(n_repeat):
            x, y = data.gen_sequence(dataset_list=datasets, seq_len=cur_seq_len, correlation='ci', fold='train')
            x, y = x.reshape(x.shape[0], -1), y.astype(int)
            x_test, y_test = data.gen_sequence(dataset_list=datasets, seq_len=cur_seq_len, correlation='iid', fold='test')
            x_test, y_test = x_test.reshape(x_test.shape[0], -1), y_test.astype(int)
            #x_test, y_test = get_remember_test_sequence(x_test, y_test)
            #x_test, y_test = get_remember_test_sequence(x, y)

            key, subkey = random.split(key, 2)
            scores_, diversity = jax.jit(inner_episode, static_argnums=[1])(subkey, update_rule, meta, x, y, x_test, y_test)
            scores += scores_
        key, subkey = random.split(key, 2)
        scores = scores / n_repeat
        meta = jax.jit(half_clone_mutate, static_argnums=[4])(subkey, meta, scores, mutation_fn=nonlocal_mutation)
        logs['mean'].append(scores.mean())
        logs['max'].append(scores.max())
        logs['diversity'].append(diversity.mean())
        print(scores.mean(), scores.max(), diversity.mean())
        if k % 100 == 0:
            save_pickle(logs, 'logs.pt')
            save_pickle(meta, 'meta_gen_5000.pt')

        if scores.max() >= 0.9:
            print('----------------------------------------------------------------------')
            break

        k += 1


