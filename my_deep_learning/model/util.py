import tensorflow as tf

def create_global_steps():
    """Creates TF ops to track and increment global training step."""
    with tf.variable_scope('global_step'):
        global_step = tf.Variable(0, name="global_step", trainable=False, dtype=tf.int32)
        increment_step = tf.assign(global_step, tf.add(global_step, 1))
    return global_step, increment_step
