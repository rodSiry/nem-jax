import jax.random as random
import jax.numpy as jnp
import jax.nn as nn

# layer norm operator
def normalize(x):
    mean = jnp.expand_dims(jnp.mean(x, 0), 0)
    std = jnp.expand_dims(jnp.std(x, 0), 0)
    return (x - mean) / (std + 1e-10)

class NEMRFAUpdateRule():
    def create_base(self, key, n_layers=3, n_input=100, n_hidden=128, n_output=10, n_state=5):
        key, subkey = random.split(key)
        h = random.normal(subkey, (n_input, n_state))
        ws = []
        rws = []
        hs = [h]

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
            w = random.normal(subkey, (n_out, n_in))

            key, subkey = random.split(key)
            rw = random.normal(subkey, (n_in, n_out))

            key, subkey = random.split(key)
            h = jnp.zeros((n_out, n_state))

            ws.append(w)
            rws.append(rw)
            hs.append(h)

        return {"w": ws, "h": hs, "rw": rws}

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
            w = random.normal(subkey, (n_in, n_out)) / jnp.sqrt(n_in)

            key, subkey = random.split(key)
            b = jnp.zeros((n_out,))

            layers.append((w, b))
        return layers


    def create_meta(self, key, n_state=5, n_act=5, n_hidden=10):
        key, subkey = random.split(key)
        forward = self._init_mlp_architecture(
            subkey, n_input=n_act + n_state, n_output=n_act, n_hidden=n_hidden
        )

        key, subkey = random.split(key)
        expand = self._init_mlp_architecture(subkey, n_input=1, n_output=n_act, n_hidden=n_hidden)

        key, subkey = random.split(key)
        collapse = self._init_mlp_architecture(subkey, n_input=n_act, n_output=1, n_hidden=n_hidden)

        key, subkey = random.split(key)
        update = self._init_mlp_architecture(
            subkey, n_input=n_act + n_act + n_state, n_output=n_state, n_hidden=n_hidden
        )

        key, subkey = random.split(key)
        to_prev = self._init_mlp_architecture(
            subkey, n_input=n_state, n_output=n_state, n_hidden=n_hidden
        )

        key, subkey = random.split(key)
        to_next = self._init_mlp_architecture(
            subkey, n_input=n_state, n_output=n_state, n_hidden=n_hidden
        )

        key, subkey = random.split(key)
        first_hidden = self._init_mlp_architecture(
            subkey, n_input=1, n_output=n_state, n_hidden=n_hidden
        )

        return {
            "forward": forward,
            "expand": expand,
            "collapse": collapse,
            "update": update,
            "to_prev": to_prev,
            "to_next": to_next,
            "first_hidden": first_hidden,
        }

    def _apply_meta_net(self, theta, x):
        y = x
        for i, (w, b) in enumerate(theta):
            y = jnp.dot(y, w)

            if i < len(theta) - 1:
                y = y + b
                y = nn.relu(y)
        return y

    def base_forward(self, meta, base, x):
        y = jnp.expand_dims(x, -1)
        y = self._apply_meta_net(meta["expand"], y)

        forward_msgs = [y]
        for i, (w, h) in enumerate(zip(base["w"], base["h"][1:])):
            y = normalize(jnp.matmul(w, y))
            forward_msgs.append(y)
            y = jnp.concatenate([y, h], -1)
            y = self._apply_meta_net(meta["forward"], y)
        pred = self._apply_meta_net(meta["collapse"], y).squeeze(-1)
        return pred, y, forward_msgs


    def update(self, meta, base, x, y):
        y_, last_act, forward_msgs = self.base_forward(meta, base, x)
        cur_backward_msg = jnp.zeros(last_act.shape)
        cur_backward_msg = cur_backward_msg.at[y].set(jnp.ones(last_act.shape[-1]))

        new_base = base

        # update inner states
        for i in range(len(base["w"])):
            j = len(base["w"]) - i - 1
            cur_forward_msg = forward_msgs[j + 1]
            update_input = jnp.concatenate(
                [base["h"][j + 1], cur_forward_msg, cur_backward_msg], -1
            )
            new_base["h"][j + 1] = jnp.clip(
                self._apply_meta_net(meta["update"], update_input), -1, 1
            )
            cur_backward_msg = normalize(jnp.matmul(base["rw"][j], cur_backward_msg))



        update_input = jnp.concatenate(
            [base["h"][0], forward_msgs[0], cur_backward_msg], -1
        )
        new_base["h"][0] = jnp.clip(self._apply_meta_net(meta["update"], update_input), -1, 1)

        # update base weights

        for i, w in enumerate(base["w"]):
            prev_state = new_base["h"][i]
            next_state = new_base["h"][i + 1]
            prv = self._apply_meta_net(meta["to_prev"], prev_state)
            prv = prv / jnp.sqrt(jnp.expand_dims(jnp.sum(prv**2, -1), -1) + 1e-10)
            nxt = self._apply_meta_net(meta["to_next"], next_state)
            nxt = nxt / jnp.sqrt(jnp.expand_dims(jnp.sum(nxt**2, -1), -1) + 1e-10)
            dw = jnp.matmul(nxt, jnp.transpose(prv))
            new_base["w"][i] = jnp.clip((w + dw), -3, 3)

        return new_base