import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import math


class LearnablePositionEncoding(layers.Layer):
    def __init__(self, d_model, dropout=0.1, max_len=10):
        super(LearnablePositionEncoding, self).__init__()
        self.dropout= layers.Dropout(dropout)
        # self.encoding = tf.Variable(tf.random.uniform((max_len, 1, d_model), -0.2, 0.2), trainable=True)
        self.encoding = tf.Variable(tf.random.uniform((max_len, d_model), -0.2, 0.2), trainable=True)

    def call(self, inputs):
        # print(f"Pos_Encoding Matrix Shape: {self.encoding[:inputs.shape[0], :].shape}")
        x = inputs + self.encoding[:inputs.shape[0], :]
        return self.dropout(x)


class TransformerBatchNormEncoderLayer(layers.Layer):

    def __init__(self, num_heads, d_model, dropout, dim_ff, activation):
        super(TransformerBatchNormEncoderLayer, self).__init__()
        self.self_attn = layers.MultiHeadAttention(num_heads, d_model, dropout=dropout)

        self.linear1 = layers.Dense(dim_ff, activation=activation)
        self.dropout = layers.Dropout(dropout)
        self.linear2 = layers.Dense(d_model, activation='linear')

        self.norm1 = layers.BatchNormalization(epsilon=1e-5)
        self.norm2 = layers.BatchNormalization(epsilon=1e-5)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, value):
        mha_out = self.self_attn(value, value)
        drop1_out = self.dropout(mha_out)
        norm1_out = self.norm1(drop1_out)
        combined_1 = norm1_out + drop1_out

        linear1_out = self.linear1(combined_1)
        drop2_out = self.dropout1(linear1_out)
        linear2_out = self.linear2(drop2_out)
        drop3_out = self.dropout2(linear2_out)

        norm2_out = self.norm2(drop3_out)

        out = norm2_out + combined_1

        return out


class TransformerBatchNormEncoderBlock(layers.Layer):

    def __init__(self, num_layers, num_heads, d_model, dropout, dim_ff, activation):
        super(TransformerBatchNormEncoderBlock, self).__init__()
        self.model = keras.Sequential()

        for _ in range(num_layers):
            self.model.add(TransformerBatchNormEncoderLayer(num_heads, d_model, dropout, dim_ff, activation))

    def call(self, inputs):
        return self.model(inputs)


class TransformerEncoderRegressor(layers.Layer):

    def __init__(self, max_len, d_model, n_heads, num_layers, dim_ff, num_classes, dropout=0.1, activation='relu', num_output_layers=1):
        super(TransformerEncoderRegressor, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_classes = num_classes

        self.input_embedding = layers.Dense(d_model, activation='linear')
        self.pos_encoding = LearnablePositionEncoding(d_model, dropout, max_len)

        self.encoder = TransformerBatchNormEncoderBlock(num_layers, n_heads, d_model, dropout, dim_ff, activation)

        self.dropout = layers.Dropout(dropout)

        self.flatten_layer = layers.Flatten()
        self.output_layers = self.build_output_module(d_model, num_output_layers, num_classes, dropout)

    def build_output_module(self, d_model, num_output_layers, num_classes, dropout=0.1):
        output = keras.Sequential()

        i = 1
        while i < num_output_layers:
            output.add(layers.Dense(d_model, activation='relu'))
            output.add(layers.Dropout(dropout))
            i += 1

        output.add(layers.Dense(num_classes, activation='linear'))

        return output

    def call(self, input):
        x = self.input_embedding(input) * math.sqrt(self.d_model)

        # print(f"shape before pos_encoding: {x.shape}") # okay
        x = self.pos_encoding(x) # problem!

        # print(f"shape before transformer: {x.shape}") # not okay
        x = self.encoder(x)
        x = self.dropout(x)

        # x = x.reshape(x.shape[0], -1)
        # print(f"shape before reshape: {x.shape}")
        # x = tf.reshape(x, (x.shape[0], -1))
        # x = tf.reshape(x, (x.shape[0], -1))
        x = self.flatten_layer(x)


        output = self.output_layers(x)

        return output


def get_model(input_shape, max_len, d_model, n_heads, num_layers, dim_ff, num_classes, dropout=0.1, activation='relu', num_output_layers=1):
    input = layers.Input(shape=input_shape)

    # print(f"shape before regressor: {input.shape}")
    regressor = TransformerEncoderRegressor(max_len, d_model, n_heads, num_layers, dim_ff, num_classes, dropout, activation, num_output_layers)

    x = regressor(input)

    return keras.Model(input, x)