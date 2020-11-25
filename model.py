import tensorflow as tf
from data import Data
import numpy as np
import logging
logging.getLogger('tensorflow').disabled = True
from statistics import mean, variance, sqrt
import matplotlib.pyplot as plt

class Model:

    def __init__(self, data: Data, combined=False, train=True):
        self.data = data
        self.vix_var, self.vix_mean = self.get_vix_params()

        #  Standardize each sample --> this makes the most sense
        self.training_input, self.testing_input = self.get_standardized_samples()
        self.training_input_vol, self.testing_input_vol = self.create_vix_input()

        # Get targets to predict change in vol
        self.training_output, self.testing_output = self.target_data_delta()
        if train:
            if combined:
                self.model = self.define_combined_lstm_model()
                self.train_combined_model()
            else:
                self.model = self.define_lstm_model()
                self.train_model()
        if combined:
            self.load_model('combined_lstm_model')
            self.eval_combined_model()
        else:
            self.load_model()
            self.eval_model()

    def get_vix_params(self):
        vix_data = self.data.training_vix_data_split

        vix_var = variance(vix_data)
        vix_mean = mean(vix_data)

        return vix_var, vix_mean

    def create_vix_input(self):
        training_input_vol = self._get_all_vol_input(self.data.training_data)
        testing_input_vol = self._get_all_vol_input(self.data.testing_data)
        return training_input_vol, testing_input_vol

    def _get_all_vol_input(self, vix_list):
        input_vol = []
        for sample in vix_list:
            initial_vol = sample[1][0]
            std_sample = self.standardize_vix_sample(initial_vol)
            input_vol.append(std_sample)

        return input_vol

    def standardize_vix_sample(self, vix_val, epsilon=1e-8):
        return (vix_val - self.vix_mean) / sqrt(self.vix_var + epsilon)

    def unstandardize_vix_sample(self, prediction, epsilon=1e-8):
        return prediction * sqrt(self.vix_var + epsilon) + - self.vix_mean

    def eval_model(self):
        output_arr = []
        actual_arr = []

        vix_actual = self.data.testing_vix_data_split[0:9]
        vix_predicted = self.data.testing_vix_data_split[0:9]

        start_ind = 1
        end_ind = 10
        for ind, sample in enumerate(self.testing_input):
            reshaped_sample = sample.reshape(1, 1, 10)
            prediction = self.model.predict(reshaped_sample)[0][0]

            output_arr.append(prediction[0])
            # actual_arr.append(actual)
            actual_arr.append(self.testing_output[ind])

            # vix_actual.append(self.data.testing_vix_data_split[start_ind] + actual)
            vix_actual.append(self.data.testing_vix_data_split[start_ind] + self.testing_output[ind])
            vix_predicted.append(self.data.testing_vix_data_split[start_ind] + prediction[0])

            start_ind += 1
            end_ind += 1

        print('DELTA:', np.corrcoef(output_arr, actual_arr))
        print('TOTAL:', np.corrcoef(vix_predicted, vix_actual))

        plt.figure()
        plt.plot(output_arr, label='Output')
        plt.plot(actual_arr, label='Actual')
        plt.ylabel('Predicted Change in Volatility', fontsize=18)
        plt.title('Predicted vs Actual Change in Volatility', fontsize=18)
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(vix_predicted, label='Output')
        plt.plot(vix_actual, label='Actual')
        plt.ylabel('Predicted Value of Volatility', fontsize=18)
        plt.title('Predicted vs Actual Volatility', fontsize=18)
        plt.legend()
        plt.show()

    def eval_combined_model(self):
        output_arr = []
        actual_arr = []

        vix_actual = self.data.testing_vix_data_split[0:9]
        vix_predicted = self.data.testing_vix_data_split[0:9]

        start_ind = 1
        end_ind = 10
        for ind, sample in enumerate(self.testing_input):
            reshaped_sample = sample.reshape(1, 1, 10)

            vix_input = self.testing_input_vol[ind]
            std_vix_input = self.standardize_vix_sample(vix_input)
            reshaped_vix_sample = std_vix_input.reshape(1, 1, 1)

            prediction = self.model.predict({"vol_input": reshaped_vix_sample, "lstm_input":reshaped_sample})[0][0]

            output_arr.append(prediction[0])
            actual_arr.append(self.testing_output[ind])

            vix_actual.append(self.data.testing_vix_data_split[start_ind] + self.testing_output[ind])
            vix_predicted.append(self.data.testing_vix_data_split[start_ind] + prediction[0])

            start_ind += 1
            end_ind += 1

        print('DELTA:', np.corrcoef(output_arr, actual_arr))
        print('TOTAL:', np.corrcoef(vix_predicted, vix_actual))

        plt.figure()
        plt.plot(output_arr, label='Output')
        plt.plot(actual_arr, label='Actual')
        plt.ylabel('Predicted Change in Volatility', fontsize=18)
        plt.title('Predicted vs Actual Change in Volatility, Combined', fontsize=18)
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(vix_predicted, label='Output')
        plt.plot(vix_actual, label='Actual')
        plt.ylabel('Predicted Value of Volatility', fontsize=18)
        plt.title('Predicted vs Actual Volatility, Combined', fontsize=18)
        plt.legend()
        plt.show()

    def define_combined_lstm_model(self):
        '''
        LSTM with additional non-time dependent input
        :return:
        '''

        # Step 1: Define LSTM input
        lstm_inputs = tf.keras.Input(shape=(None, 10), name="lstm_input")
        lstm_layer = tf.keras.layers.LSTM(10, return_sequences=True, return_state=True, activation='relu')
        lstm_outputs, state_h, state_c = lstm_layer(lstm_inputs)

        # Step 2: Define Vol input
        init_vol_inputs = tf.keras.Input(shape=(None, 1), name="vol_input")

        # Concatenate LSTM output with the initial volitility
        concat_layer = tf.keras.layers.concatenate([init_vol_inputs, lstm_outputs])

        # Output a single numerical value
        output_layer = tf.keras.layers.Dense(1)
        network_output = output_layer(concat_layer)
        lstm_model = tf.keras.Model(inputs=[init_vol_inputs, lstm_inputs], outputs=network_output)

        # Compile model
        lstm_model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
        print(lstm_model.summary())

        return lstm_model

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

    def load_model(self, name='lstm_model'):
        self.model = tf.keras.models.load_model(name)
        print('Loaded model:', name)

    def train_model(self):
        reshaped_input = np.asarray(self.training_input).reshape(len(self.training_input), 1, 10)
        self.model.fit(reshaped_input,  np.asarray(self.training_output), epochs=50, validation_split=0.2, batch_size=5)
        self.model.save('lstm_model')

    def train_combined_model(self):
        reshaped_lstm_input = np.asarray(self.training_input).reshape(len(self.training_input), 1, 10)
        reshaped_vol_input = np.asarray(self.training_input_vol).reshape(len(self.training_input_vol), 1, 1)

        self.model.fit({"vol_input": reshaped_vol_input, "lstm_input": reshaped_lstm_input}, np.asarray(self.training_output), epochs=50, validation_split=0.2, batch_size=5)
        self.model.save('combined_lstm_model')

    # todo: preprocess this too?
    def target_data_delta(self):
        '''
        Defines target output as the change in vol over the given time period
        :return:
        '''
        training_targets = []
        for sample in self.data.training_data:
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
