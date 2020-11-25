import tensorflow as tf
from data import Data
import numpy as np
import tensorflow_transform as tft
# Supress deprecation warnings
import logging
logging.getLogger('tensorflow').disabled = True
# tf.compat.v1.disable_eager_execution()
from statistics import mean, variance, sqrt
import matplotlib.pyplot as plt

class Model:

    def __init__(self, data: Data):
        self.data = data

        # Need to preprocess the data
        # Several options:
        # 1) Normalize from 0:1
        # 2) Normalize within each sample (more like 0 mean)
        # 3) Standardize

        # 4) Standardize each sample --> this makes the most sense
        self.training_input, self.testing_input = self.get_standardized_samples()

        # Get targets
        # Several options:
        #   1) Predict change in vol
        self.training_output, self.testing_output = self.target_data_delta()

        #   2) seq2seq (basically a volitility model)
        # self.training_output, self.testing_output = self.target_data_all()
        #
        # self.model = self.define_lstm_model()
        # self.train_model()

        self.load_model()
        self.eval_model()

    def eval_model(self):
        output_arr = []
        actual_arr = []

        vix_actual = self.data.testing_vix_data_split[0:9]
        vix_predicted = self.data.testing_vix_data_split[0:9]
        start_ind = 1
        end_ind = 10
        for ind, sample in enumerate(self.testing_input):
            # print(sample)
            reshaped_sample = sample.reshape(1, 1, 10)
            prediction = self.model.predict(reshaped_sample)[0][0]
            # print(prediction, self.testing_output[ind])
            output_arr.append(prediction)
            actual_arr.append(self.testing_output[ind])

            # vix_actual.append(self.data.testing_vix_data_split[end_ind])
            vix_actual.append(self.data.testing_vix_data_split[start_ind] + self.testing_output[ind])
            vix_predicted.append(self.data.testing_vix_data_split[start_ind] + prediction[0])
            start_ind += 1
            end_ind += 1

        plt.figure()
        plt.plot(output_arr, label='Output')
        plt.plot(actual_arr, label='Actual')
        plt.ylabel('Predicted Change in Volitility')
        plt.legend()
        plt.show()

        print(vix_actual)
        print(vix_predicted)

        plt.figure()
        plt.plot(vix_predicted, label='Output')
        plt.plot(vix_actual, label='Actual')
        plt.ylabel('Predicted Value of Volitility')
        plt.title('Predicted vs Actual Volitility')
        plt.legend()
        plt.show()


    def define_lstm_model(self):
        '''
        This is set up for a single output value
        :return:
        '''

        # Step 1: Define input
        lstm_inputs = tf.keras.Input(shape=(None, 10))
        lstm_layer = tf.keras.layers.LSTM(10, return_sequences=True, return_state=True, activation='relu')
        lstm_outputs, state_h, state_c = lstm_layer(lstm_inputs)

        # Output a single numerical value
        output_layer = tf.keras.layers.Dense(1)
        network_output = output_layer(lstm_outputs)
        lstm_model = tf.keras.Model(inputs=lstm_inputs, outputs=network_output)

        # Compile model
        lstm_model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
        print(lstm_model.summary())
        return lstm_model

    def load_model(self):
        self.model = tf.keras.models.load_model('lstm_model')
        print('Loaded model')

    def train_model(self):
        reshaped_input = np.asarray(self.training_input).reshape(len(self.training_input), 1, 10)
        history = self.model.fit(reshaped_input,  np.asarray(self.training_output), epochs=50, validation_split=0.2, batch_size=5)

        self.model.save('lstm_model')


    # todo: preprocess this too?
    def target_data_delta(self):
        '''
        Defines target output as the change in vol over the given time period
        :return:
        '''
        training_targets = []
        for sample in self.data.training_data:
            v_diff = self._vol_diff(sample)
            # print(v_diff)
            # print(v_diff + sample[1][0], sample[1][-1])
            training_targets.append(self._vol_diff(sample))

        testing_targets = []
        for sample in self.data.testing_data:
            testing_targets.append(self._vol_diff(sample))

        return training_targets, testing_targets

    def target_data_all(self):
        training_targets = []
        for sample in self.data.training_data:
            training_targets.append(sample[1])

        testing_targets = []
        for sample in self.data.testing_data:
            testing_targets.append(sample[1])

        return training_targets, testing_targets

    def get_standardized_samples(self):
        training_samples = []
        for sample in self.data.training_data:
            std_sample = self._standardize_sample(sample[0])
            training_samples.append(std_sample)

        testing_samples = []
        for sample in self.data.testing_data:
            std_sample = self._standardize_sample(sample[0])
            testing_samples.append(std_sample)

        return training_samples, testing_samples

    @staticmethod
    def _standardize_sample(sample, epsilon=1e-8):
        mean_val = mean(sample)
        var = variance(sample)
        standardized_sample = (sample - mean_val) / sqrt(var + epsilon)
        return standardized_sample

    @staticmethod
    def _vol_diff(sample):
        return sample[1][-1] - sample[1][0]
