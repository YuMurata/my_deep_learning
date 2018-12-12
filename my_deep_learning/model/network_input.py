import tensorflow as tf
from my_deep_learning.model.util import create_global_steps
class NetworkInputCreator:
    def __init__(self, scope:str=''):       
        self.scope =  scope
        with tf.variable_scope(self.scope):
            self.global_step, self.increment_step = create_global_steps()
                                                  
   
    @staticmethod
    def create_visual_input(camera_parameters, name):
        """
        Creates image input op.
        :param camera_parameters: Parameters for visual observation from BrainInfo.
        :param name: Desired name of input op.
        :return: input op.
        """
        o_size_h = camera_parameters['height']
        o_size_w = camera_parameters['width']
        bw = camera_parameters['blackAndWhite']

        if bw:
            c_channels = 1
        else:
            c_channels = 3

        visual_in = tf.placeholder(shape=[None, o_size_h, o_size_w, c_channels], dtype=tf.float32,
                                   name=name)
        return visual_in

    def create_vector_input(self, normalize:bool, name:str, vector_shape:tuple, mean_shape:tuple):
        """
        Creates ops for vector observation input.
        :param name: Name of the placeholder op.
        :param vector_size: Size of stacked vector observation.
        :return:
        """
        with tf.variable_scope(name):
            vector_in = tf.placeholder(shape=vector_shape, dtype=tf.float32,
                                                    name=name + '_placeholder')
            if normalize:
                running_mean = tf.get_variable('running_mean', shape=mean_shape,
                                                    trainable=False, dtype=tf.float32,
                                                    initializer=tf.zeros_initializer())
                running_variance = tf.get_variable('running_variance', shape=mean_shape,
                                                        trainable=False,
                                                        dtype=tf.float32,
                                                        initializer=tf.ones_initializer())
                update_mean, update_variance = self.create_normalizer_update(vector_in, running_mean, running_variance)

                normalized_vector = tf.clip_by_value((vector_in - update_mean) / tf.sqrt(
                    update_variance / (tf.cast(self.global_step, tf.float32) + 1)), -5, 5,
                                                        name='normalized_vector')
                return normalized_vector
            else:
                return vector_in

    def create_normalizer_update(self, vector_input, running_mean, running_variance):
        with tf.variable_scope('normalize'):
            mean_current_observation = tf.reduce_mean(vector_input, axis=0)
            new_mean = running_mean + (mean_current_observation - running_mean) / \
                    tf.cast(tf.add(self.global_step, 1), tf.float32)
            new_variance = running_variance + (mean_current_observation - new_mean) * \
                        (mean_current_observation - running_mean)
            update_mean = tf.assign(running_mean, new_mean)
            update_variance = tf.assign(running_variance, new_variance)
        return update_mean, update_variance