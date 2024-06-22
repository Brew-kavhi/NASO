import tensorflow as tf


def compute_mean_and_std(ds):
    count = 0.0
    mean_targets = 0.0
    M2_targets = 0.0

    for inputs, targets in ds:
        batch_size = tf.shape(targets)[0]

        # Calculate batch mean and variance for targets
        batch_mean_targets = tf.reduce_mean(targets, axis=0)
        batch_var_targets = tf.reduce_mean((targets - batch_mean_targets) ** 2, axis=0)

        # Update global mean and variance for targets using Welford's method
        delta_targets = batch_mean_targets - mean_targets
        mean_targets += (
            delta_targets
            * tf.cast(batch_size, tf.double)
            / (count + tf.cast(batch_size, tf.double))
        )
        M2_targets += batch_var_targets * tf.cast(
            batch_size, tf.double
        ) + delta_targets**2 * tf.cast(count, tf.double) * tf.cast(
            batch_size, tf.double
        ) / (
            count + tf.cast(batch_size, tf.double)
        )

        count += tf.cast(batch_size, tf.double)

    std_targets = tf.sqrt(M2_targets / tf.cast(count, tf.double))
    return mean_targets, std_targets


# Define a function to normalize the dataset using target mean and std
def z_normalize_ds(ds, mean_targets, std_targets):
    def normalize(inputs, targets):
        norm_targets = (targets - mean_targets) / std_targets
        return inputs, norm_targets

    return ds.map(lambda inputs, targets: normalize(inputs, targets))


def z_scaler(ds):
    mean, std = compute_mean_and_std(ds)
    return z_normalize_ds(ds, min, max)


def compute_min_and_max(ds):
    min_val = float("inf")
    max_val = float("-inf")

    for batch in ds:
        batch_min = tf.reduce_min(batch, axis=0)
        batch_max = tf.reduce_max(batch, axis=0)

        min_val = tf.minimum(min_val, batch_min)
        max_val = tf.maximum(max_val, batch_max)

    return min_val, max_val


def min_max_normalizer(ds, min_val, max_val):
    def scale(x):
        return (x - min_val) / (max_val - min_val)

    return ds.map(lambda x: scale(x))


def min_max_scaler(ds):
    min, max = compute_min_and_max(ds)
    return min_max_normalizer(ds, min, max)


def mean_normalize(ds, mean, min_val, max_val):
    def normalize(x):
        return (x - mean) / (max_val - min_val)

    return ds.map(lambda x: normalize(x))


def mean_scaler(ds):
    mean, _ = compute_mean_and_std(ds)
    min, max = compute_min_and_max(ds)
    return mean_normalize(ds, mean, min, max)
