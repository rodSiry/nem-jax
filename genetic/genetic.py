import jax.random as random
import jax.numpy as jnp
import jax
def compute_novelty(meta, n_nearest=10):
    def dist_fn(t1, t2):
        t1 = (t1 - t1.mean()) / (t1.std() + 1e-10)
        t2 = (t2 - t2.mean()) / (t2.std() + 1e-10)

        t1 = jnp.reshape(t1, (t1.shape[0], -1))
        t2 = jnp.reshape(t2, (t2.shape[0], -1))

        t1 = jnp.expand_dims(t1, 0)
        t2 = jnp.expand_dims(t2, 1)

        return jnp.abs(t1 - t2).mean(-1)

    dist_tree = jax.tree_map(dist_fn, meta, meta)
    res = jax.tree_util.tree_reduce(lambda x, y: x + y, dist_tree, 0)
    res = jax.vmap(jnp.sort)(res)
    return res[:, :n_nearest].mean(-1)


def tree_of_keys(key, tree):
    treedef = jax.tree_util.tree_structure(tree)
    keys = random.split(key, treedef.num_leaves)
    return jax.tree_util.tree_unflatten(treedef, keys)

def gaussian_mutation(key, tree, eps=0.01):
    keys_tree = tree_of_keys(key, tree)
    new_tree = jax.tree_map(lambda k, t: jnp.clip(t + eps * random.normal(k, t.shape), -3, 3), keys_tree, tree)
    return new_tree

def nonlocal_mutation(key, tree, eps=0.01):
    key, subkey = random.split(key, 2)
    keys_tree_1 = tree_of_keys(subkey, tree)
    key, subkey = random.split(key, 2)
    keys_tree_2 = tree_of_keys(subkey, tree)

    def map_function(k1, k2, t):
        n = random.normal(k1, t.shape)
        b = random.bernoulli(k2, 0.01, t.shape)
        return jnp.clip(b * n + (1 - b) * t, -3, 3)
    new_tree = jax.tree_map(map_function, keys_tree_1, keys_tree_2, tree)
    return new_tree

def half_clone_mutate(key, meta, scores, threshold=500, mutation_fn=nonlocal_mutation):
    indices = jnp.flip(jnp.argsort(scores))
    selected_meta = jax.tree_map(lambda x : x[indices][:threshold], meta)
    mutated_meta = mutation_fn(key, selected_meta)
    return jax.tree_map(lambda x, y : jnp.concatenate([x, y], 0), mutated_meta, selected_meta)

