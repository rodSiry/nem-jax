import jax
import jax.nn as nn
import jax.numpy as jnp
import jax.random as random
from loaders import SequenceGenerator
from random import randint

def compute_novelty(meta, n_nearest=10):
    def dist_fn(t1, t2):

        t1 = jnp.reshape(t1, (t1.shape[0], -1))
        t2 = jnp.reshape(t2, (t2.shape[0], -1))

        t1 = jnp.expand_dims(t1, 0)
        t2 = jnp.expand_dims(t2, 1)

        return jnp.abs(t1 - t2).mean(-1)

    dist_tree = jax.tree_map(dist_fn, meta, meta)
    res = jax.tree_util.tree_reduce(lambda x, y: x + y, dist_tree, 0)
    return res.mean(-1)

def get_remember_test_sequence(x, y):
    new_x = jnp.zeros(x.shape)
    new_y = jnp.zeros(y.shape).astype(int)
    for i in range(x.shape[0]):
        r = randint(0, i)
        new_x = new_x.at[i].set(x[r])
        new_y = new_y.at[i].set(y[r])
    return new_x, new_y

def ga_iteration(key, meta, x, y, x_test, y_test):
    vmap_create_base = jax.vmap(create_base, in_axes=[0, None, None, None, None, None])
    vmap_update = jax.vmap(update, in_axes=[0, 0, None, None])
    vmap_inference = jax.vmap(base_forward, in_axes=[0, 0, None])

    key, subkey = random.split(key, 2)
    keys = random.split(subkey, 1000)
    base = vmap_create_base(keys, 256, 10, 128, 3, 3)

    def inner_step(acc, input_data):
        x = input_data['x']
        y = input_data['y']
        x_test = input_data['x_test']
        y_test = input_data['y_test']
        new_acc, e = vmap_update(meta, acc, x, y)
        y_test_ = jnp.argmax(vmap_inference(meta, new_acc, x_test)[0], -1)
        e = (y_test_ == y_test)
        return new_acc, e

    input_data = {'x':x, 'y':y, 'x_test':x_test, 'y_test':y_test}
    base, scores = jax.lax.scan(inner_step, base, input_data)
    scores = scores.mean(0)

    diversity = compute_novelty(meta)

    key, subkey = random.split(key, 2)
    new_meta = half_clone_mutate(subkey, meta, 0.01 * diversity)
    return new_meta, scores, diversity



def tree_of_keys(key, tree):
    treedef = jax.tree_util.tree_structure(tree)
    keys = random.split(key, treedef.num_leaves)
    return jax.tree_util.tree_unflatten(treedef, keys)

def gaussian_mutation(key, tree, eps=0.01):
    keys_tree = tree_of_keys(key, tree)
    new_tree = jax.tree_map(lambda k, t: t + eps * random.normal(k, t.shape), keys_tree, tree)
    return new_tree

def nonlocal_mutation(key, tree, eps=0.01):
    key, subkey = random.split(key, 2)
    keys_tree_1 = tree_of_keys(subkey, tree)
    key, subkey = random.split(key, 2)
    keys_tree_2 = tree_of_keys(subkey, tree)

    def map_function(k1, k2, t):
        n = random.normal(k1, t.shape)
        b = random.bernoulli(k2, 0.01, t.shape)
        return b * n + (1 - b) * t
    new_tree = jax.tree_map(map_function, keys_tree_1, keys_tree_2, tree)
    return new_tree

def half_clone_mutate(key, meta, scores, threshold=500, mutation_fn=gaussian_mutation):
    indices = jnp.flip(jnp.argsort(scores))
    selected_meta = jax.tree_map(lambda x : x[indices][:threshold], meta)
    mutated_meta = mutation_fn(key, selected_meta)
    return jax.tree_map(lambda x, y : jnp.concatenate([x, y], 0), mutated_meta, selected_meta)


def normalize(x):
    mean, std = x.mean(), x.std()
    return (x - mean) / (std + 1e-10)
def create_base(key, n_in, n_out, n_hidden, n_layers, n_state=3):
    h = []
    rw = []
    for i in range(n_layers):
        in_ = n_in if i==0 else n_hidden
        out_ = n_out if i == n_layers -1 else n_hidden
        subkey, key = random.split(key)
        h.append(random.normal(subkey, (in_, out_, n_state)))

        subkey, key = random.split(key)
        if i<n_layers - 1:
            rw.append(random.normal(subkey, (out_, n_out)) / jnp.sqrt(n_out))
        else:
            rw.append(jnp.eye(n_out) / jnp.sqrt(n_out))
    return {'h':h, 'rw':rw}

def init_mlp_architecture(key, n_layers=2, n_input=1, n_hidden=10, n_output=1):
    layers = []

    for i in range(n_layers):
        if i == 0:
            n_in = n_input
            n_out = n_hidden
        elif i == n_layers - 1:
            n_in = n_hidden
            n_out = n_output
        else:
            n_in = n_hidden
            n_out = n_hidden

        key, subkey = random.split(key)
        w = random.normal(subkey, (n_in, n_out))

        key, subkey = random.split(key)
        b = random.normal(subkey, (n_out,))

        layers.append((w, b))

    return layers
def create_meta(key, n_state=3):
    key, subkey = random.split(key, 2)
    to_w = init_mlp_architecture(subkey, n_layers=2, n_input=3, n_hidden=2, n_output=1)
    key, subkey = random.split(key, 2)
    update = init_mlp_architecture(subkey, n_layers=2, n_input=3+3, n_hidden=2, n_output=3)
    return {'to_w':to_w, 'update':update}

def apply_meta_net(theta, x):
    y = x
    for i, (w, b) in enumerate(theta):
        y = jnp.dot(y, w)
        y = y + b
        if i < len(theta) - 1:
            y = nn.relu(y)
    return y
def base_forward(meta, base, x):
    y = x
    prev_act = []
    next_act = []
    for i,h in enumerate(base['h']):
        prev_act.append(y)
        w = apply_meta_net(meta['to_w'], h).squeeze(-1)
        y = jnp.matmul(y, w) / jnp.sqrt(w.shape[0])
        next_act.append(y)
        if i < len(base['h']) - 1:
            y = nn.relu(y)
    return y, prev_act, next_act

def update(meta, base, x, Y):
    y, prev_act, next_act = base_forward(meta, base, x)
    e = (y - Y)
    errors = [e] * len(base['rw'])

    backward_msgs = jax.tree_map(jnp.matmul, base['rw'], errors)

    def update_state(p_act, n_act, b_msg, h):
        e_p_act = jnp.repeat(jnp.expand_dims(p_act, -1), h.shape[1], -1)
        e_n_act = jnp.repeat(jnp.expand_dims(n_act,  0), h.shape[0],  0)
        e_b_msg = jnp.repeat(jnp.expand_dims(b_msg,  0), h.shape[0],  0)
        update_data = jnp.concatenate([jnp.stack([e_p_act, e_n_act, e_b_msg], -1), h], -1)

        new_h = apply_meta_net(meta['update'], update_data)
        new_h = jnp.clip(new_h, -1, 1)
        return new_h

    new_h = jax.tree_map(update_state, prev_act, next_act, backward_msgs, base['h'])

    return {'h':new_h, 'rw':base['rw']}, e


vmap_create_meta = jax.vmap(create_meta, in_axes=[0, None])

data = SequenceGenerator()

key = random.PRNGKey(0)


key, subkey = random.split(key, 2)
keys = random.split(subkey, 1000)
meta = vmap_create_meta(keys, 3)

for _ in range(1000):
    x, y = data.gen_sequence(dataset_list=['cifar10'], seq_len=10)
    x = x.reshape(x.shape[0], -1)
    x_test, y_test = get_remember_test_sequence(x, y)
    key, subkey = random.split(key, 2)
    meta, scores, diversity = jax.jit(ga_iteration)(key, meta, x, y, x_test, y_test)
    print(scores.mean(), scores.max(), diversity.mean())