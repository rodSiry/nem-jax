import jax
import jax.nn as nn
import jax.numpy as jnp
import jax.random as random
from random import randint

def normalize(x):
    mean, std = x.mean(), x.std()
    return (x - mean) / (std + 1e-10)

class SynapseUpdateRule():
    def create_base(self, key, n_in, n_out, n_hidden, n_layers, n_state=3):
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

    def _init_mlp_architecture(self, key, n_layers=2, n_input=1, n_hidden=10, n_output=1):
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
    def create_meta(self, key, n_state=3):
        key, subkey = random.split(key, 2)
        to_w = self._init_mlp_architecture(subkey, n_layers=2, n_input=3, n_hidden=2, n_output=1)
        key, subkey = random.split(key, 2)
        update = self._init_mlp_architecture(subkey, n_layers=2, n_input=3+3, n_hidden=2, n_output=3)
        return {'to_w':to_w, 'update':update}

    def _apply_meta_net(self, theta, x):
        y = x
        for i, (w, b) in enumerate(theta):
            y = jnp.dot(y, w)
            y = y + b
            if i < len(theta) - 1:
                y = nn.relu(y)
        return y
    def base_forward(self, meta, base, x):
        y = x
        prev_act = []
        next_act = []
        for i,h in enumerate(base['h']):
            prev_act.append(y)
            w = self._apply_meta_net(meta['to_w'], h).squeeze(-1)
            y = jnp.matmul(y, w) / jnp.sqrt(w.shape[0])
            next_act.append(y)
            if i < len(base['h']) - 1:
                y = nn.relu(y)
        return y, prev_act, next_act

    def update(self, meta, base, x, Y):
        y, prev_act, next_act = self.base_forward(meta, base, x)
        e = (y - Y)
        errors = [e] * len(base['rw'])

        backward_msgs = jax.tree_map(jnp.matmul, base['rw'], errors)

        def update_state(p_act, n_act, b_msg, h):
            e_p_act = jnp.repeat(jnp.expand_dims(p_act, -1), h.shape[1], -1)
            e_n_act = jnp.repeat(jnp.expand_dims(n_act,  0), h.shape[0],  0)
            e_b_msg = jnp.repeat(jnp.expand_dims(b_msg,  0), h.shape[0],  0)
            update_data = jnp.concatenate([jnp.stack([e_p_act, e_n_act, e_b_msg], -1), h], -1)

            new_h = self._apply_meta_net(meta['update'], update_data)
            new_h = jnp.clip(new_h, -1, 1)
            return new_h

        new_h = jax.tree_map(update_state, prev_act, next_act, backward_msgs, base['h'])

        return {'h':new_h, 'rw':base['rw']}, e