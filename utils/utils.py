import pickle
import numpy as np
import imageio
import jax.numpy as jnp
import jax


def save_pickle(obj, filename):
    F = open(filename, "wb")
    pickle.dump(obj, F)
    F.close()


def load_pickle(filename):
    F = open(filename, "rb")
    obj = pickle.load(F)
    F.close()
    return obj

def convolve(x, N=10):
    return np.convolve(x, np.ones(N) / N, mode="valid")

def filters_to_grids(filters):
    seq_len = filters.shape[0]
    frames = filters.reshape(seq_len, 16, 8, 16, 16, 1)
    frames = jnp.transpose(frames, [0, 1, 3, 2, 4, 5])
    frames = frames.reshape(seq_len, 16 * 16, 8 * 16, 1)


    def img_normalize(img):
        return (img - jnp.min(img)) / (jnp.max(img) - jnp.min(img))
    frames = jax.vmap(img_normalize)(frames)

    #frames = (frames - jnp.min(frames)) / (jnp.max(frames) - jnp.min(frames))
    frames = jnp.repeat(frames, 3, -1)
    return frames

def filters_to_one_filter_grid(filters):
    seq_len = filters.shape[0]
    frames = filters[:, 0]
    frames = frames.reshape(seq_len, 16, 16, 1)
    def img_normalize(img):
        return (img - jnp.min(img)) / (jnp.max(img) - jnp.min(img))
    frames = jax.vmap(img_normalize)(frames)
    frames = jnp.repeat(frames, 3, -1)
    return frames

def make_gif(filepath, frames):
    """
    with imageio.get_writer(filepath, mode='I') as writer:
        for frame in frames:
            writer.append_data(frame)
    """
    frames = [frames[i] for i in range(frames.shape[0])]
    imageio.mimsave(filepath, frames, fps=200)
