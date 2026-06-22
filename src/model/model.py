import keras
from keras.models import Model, Sequential
from keras.layers import Input, Dense, LSTM, GRU, Dropout, Conv1D, MaxPooling1D, Flatten, Concatenate, BatchNormalization, LayerNormalization, GlobalAveragePooling1D, Reshape, Layer, AdditiveAttention
from keras.regularizers import l2
from keras.metrics import AUC, Precision, Recall, F1Score, FalsePositives
from tensorflow import stop_gradient


@keras.saving.register_keras_serializable()
class StopGradient(Layer):
    def call(self, x):
        return stop_gradient(x)

    def get_config(self):
        return super().get_config()


class IDSModelFactory:
	@staticmethod
	def create_model(
		window_size=80, num_features=80, num_classes=15,
		num_groups=None, head_depth="standard",
		conv_l2=0.0001,
		use_first_bn=True,
		bn_momentum=0.97,
		rnn_type="lstm",
		rnn_units=128,
		rnn_layers=1,
	):
		input_dim_ae = num_features
		input_dim_cnn = (window_size, num_features)
		input_dim_lstm = (window_size, num_features)

		ae_input = Input(shape=(input_dim_ae,), name='ae_input')
		ae_encoded = Dense(48, activation='relu', name='ae_encoder')(ae_input)
		ae_bottleneck = Dense(24, activation='relu', name='ae_bottleneck')(ae_encoded)
		ae_decoded = Dense(48, activation='relu', name='ae_decoder')(ae_bottleneck)
		ae_output = Dense(input_dim_ae, activation='sigmoid', name='ae_output')(ae_decoded)

		ae_features = StopGradient(name='ae_features')(ae_bottleneck)

		cnn_input = Input(shape=input_dim_cnn, name='cnn_input')
		cnn_conv1 = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(conv_l2))(cnn_input)
		if use_first_bn:
			cnn_conv1 = BatchNormalization(momentum=bn_momentum, epsilon=1e-5)(cnn_conv1)
		cnn_pool1 = MaxPooling1D(pool_size=2, padding='same')(cnn_conv1)
		cnn_conv2 = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(conv_l2))(cnn_pool1)
		cnn_bn2 = BatchNormalization(momentum=bn_momentum, epsilon=1e-5)(cnn_conv2)
		cnn_pool2 = MaxPooling1D(pool_size=2, padding='same')(cnn_bn2)
		cnn_global_pooled = GlobalAveragePooling1D(name='cnn_global_pool')(cnn_pool2)
		cnn_dropout = Dropout(0.2)(cnn_global_pooled)

		cnn_features = cnn_dropout

		lstm_input = Input(shape=input_dim_lstm, name='lstm_input')
		lstm_lm = LayerNormalization(epsilon=1e-3, center=True, scale=True)(lstm_input)
		rnn_layer = LSTM if rnn_type == "lstm" else GRU
		rnn_out = lstm_lm
		for i in range(rnn_layers):
			return_seq = i < rnn_layers - 1
			rnn_out = rnn_layer(rnn_units, return_sequences=return_seq, dropout=0.2)(rnn_out)
		lstm_dropout = Dropout(0.3)(rnn_out)

		lstm_features = lstm_dropout

		combined = Concatenate(name='feature_fusion')([
			ae_features,
			cnn_features,
			lstm_features
		])

		# === CLASSIFICATION HEAD (configurable depth) ===
		if head_depth == "attention":
			combined_3d = Reshape((1, -1))(combined)
			attention = AdditiveAttention(name='self_attention')([combined_3d, combined_3d])
			attention = Dropout(0.2)(attention)
			fused = LayerNormalization()(combined_3d + attention)
			fused = Flatten()(fused)
			
			dense = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(fused)
			dense_bn = BatchNormalization()(dense)
			dense_dropout = Dropout(0.3)(dense_bn)
			classification_output = Dense(num_classes, activation='softmax', name='classification')(dense_dropout)
		elif head_depth == "deep":
			dense = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(combined)
			dense_bn = BatchNormalization()(dense)
			dense_dropout = Dropout(0.3)(dense_bn)
			dense2 = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(dense_dropout)
			dense2_bn = BatchNormalization()(dense2)
			dense2_dropout = Dropout(0.25)(dense2_bn)
			classification_output = Dense(num_classes, activation='softmax', name='classification')(dense2_dropout)
		elif head_depth == "shallow":
			dense = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(combined)
			dense_bn = BatchNormalization()(dense)
			dense_dropout = Dropout(0.3)(dense_bn)
			classification_output = Dense(num_classes, activation='softmax', name='classification')(dense_dropout)
		else:  # "standard"
			dense = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(combined)
			dense_bn = BatchNormalization()(dense)
			dense_dropout = Dropout(0.3)(dense_bn)
			classification_output = Dense(num_classes, activation='softmax', name='classification')(dense_dropout)

		model = Model(
			inputs={
				'ae_input': ae_input,
				'cnn_input': cnn_input,
				'lstm_input': lstm_input
			},
			outputs={
				'classification': classification_output,
				'reconstruction': ae_output
			}
		)

		return model
	
	@staticmethod
	def model_summary(model):
		print("=" * 80)
		print("HYBRID IDS MODEL ARCHITECTURE")
		print("=" * 80)
		print(f"CNN Branch: 2x (Conv1D(32) → BatchNorm → MaxPool)")
		print(f"LSTM Branch: LSTM(128) with LayerNorm")
		print(f"Fusion Layer: Combined features ({32+32+128} dimensions)")
		print("=" * 80)
		model.summary()

	def save_model(model, name:str):
		return model.save(name)

	
		
if __name__ == "__main__":
	model = IDSModelFactory.create_model(num_classes=10)
	model.compile(
		optimizer="adam", 
		metrics=["accuracy", Precision(), Recall(), F1Score(), AUC(), FalsePositives()],
		loss={
			"classification": "categorical_crossentropy",
			"reconstruction": "mse"
		},
		loss_weights={
			"classification": 1.0,
			"reconstruction": 0.3,
		}
	)

	IDSModelFactory.model_summary(model)
