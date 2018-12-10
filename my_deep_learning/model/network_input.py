import tensorflow as tf

class NetworkInputCreator:
    def __init__(self, normalize:bool, name:str, vector_shape, mean_shape):        
        self.name = name
        self.normalize = normalize
        self.vector_shape = vector_shape
        self.mean_shape = mean_shape
        with tf.variable_scope('input_creator'):
            with tf.variable_scope(self.name):
                self.global_step, self.increment_step = self.create_global_steps()
                self.vector_in = tf.placeholder(shape=vector_shape, dtype=tf.float32,
                                                name=self.name + '_placeholder')
                self.vector_in = self.create_vector_input()                                        

    @staticmethod
    def create_global_steps():
        """Creates TF ops to track and increment global training step."""
        global_step = tf.Variable(0, name="global_step", trainable=False, dtype=tf.int32)
        increment_step = tf.assign(global_step, tf.add(global_step, 1))
        return global_step, increment_step

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

    def create_vector_input(self):
        """
        Creates ops for vector observation input.
        :param name: Name of the placeholder op.
        :param vector_size: Size of stacked vector observation.
        :return:
        """
        if self.normalize:
            self.running_mean = tf.get_variable('running_mean', shape=self.mean_shape,
                                                trainable=False, dtype=tf.float32,
                                                initializer=tf.zeros_initializer())
            self.running_variance = tf.get_variable('running_variance', shape=self.mean_shape,
                                                    trainable=False,
                                                    dtype=tf.float32,
                                                    initializer=tf.ones_initializer())
            self.update_mean, self.update_variance = self.create_normalizer_update(self.vector_in)

            self.normalized_vector = tf.clip_by_value((self.vector_in - self.running_mean) / tf.sqrt(
                self.running_variance / (tf.cast(self.global_step, tf.float32) + 1)), -5, 5,
                                                     name='normalized_vector')
            return self.normalized_vector
        else:
            return self.vector_in

    def create_normalizer_update(self, vector_input):
        mean_current_observation = tf.reduce_mean(vector_input, axis=0)
        new_mean = self.running_mean + (mean_current_observation - self.running_mean) / \
                   tf.cast(tf.add(self.global_step, 1), tf.float32)
        new_variance = self.running_variance + (mean_current_observation - new_mean) * \
                       (mean_current_observation - self.running_mean)
        update_mean = tf.assign(self.running_mean, new_mean)
        update_variance = tf.assign(self.running_variance, new_variance)
        return update_mean, update_variance