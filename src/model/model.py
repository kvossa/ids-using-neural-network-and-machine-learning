from keras.models import Model, Sequential
from keras.layers import Input, Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten, concatenate, BatchNormalization, LayerNormalization, GlobalAveragePooling1D, Reshape
from keras.regularizers import l2

# import mlflow
# mlflow.tensorflow.log_model(model, "model")

# from multiprocessing import Pool
# pool = Pool(processes=4)

# import smtplib

class IDSModelFactory:
	@staticmethod
	def create_model(input_dim_lstm=(100,10), input_dim_cnn=(100,10), input_dim_ae=20, num_classes=15):
		ae_input = Input(shape=(input_dim_ae,), name='ae_input')
		ae_encoded = Dense(64, activation='relu', name='ae_encoder')(ae_input)
		ae_bottleneck = Dense(32, activation='relu', name='ae_bottleneck')(ae_encoded)
		ae_decoded = Dense(64, activation='relu', name='ae_decoder')(ae_bottleneck)
		ae_output = Dense(input_dim_ae, activation='sigmoid', name='ae_output')(ae_decoded)

		ae_features = ae_bottleneck

		cnn_input = Input(shape=input_dim_cnn, name='cnn_input')
		cnn_conv1 = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(cnn_input)
		cnn_bn1 = BatchNormalization(momentum=0.97, epsilon=1e-5)(cnn_conv1)
		cnn_pool1 = MaxPooling1D(pool_size=2)(cnn_bn1)
		cnn_conv2 = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(cnn_pool1)
		cnn_bn2 = BatchNormalization(momentum=0.97, epsilon=1e-5)(cnn_conv2)
		cnn_pool2 = MaxPooling1D(pool_size=2)(cnn_bn2)
		cnn_global_pooled = GlobalAveragePooling1D(name='cnn_global_pool')(cnn_pool2)
		cnn_dropout = Dropout(0.2)(cnn_global_pooled)

		cnn_features = cnn_dropout

		lstm_input = Input(shape=input_dim_lstm, name='lstm_input')
		lstm_lm = LayerNormalization(epsilon=1e-3, center=True, scale=True)(lstm_input)
		lstm_out = LSTM(128, return_sequences=False, dropout=0.2)(lstm_lm)
		lstm_dropout = Dropout(0.3)(lstm_out)

		lstm_fetures = lstm_dropout

		combined = concatenate(name='feature_fusion')([
			ae_feautures,
			cnn_features,
			lstm_features
		])

		dense = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(combined)
		dense_bn = BatchNormalization()(dense)
		dense_dropout = Dropout(0.3)(dense_bn)

		output = Dense(num_classes, activation='softmax', name='classification_output')(dense_dropout)

		model = Model(
			inputs={
				'ae_input': ae_input,
				'cnn_input': cnn_input,
				'lstm_input': lstm_input
			},
			outputs={
				'classification': output,
				'reconstruction': ae_output
			}
		)

		return model
	
	@staticmethod
	def model_summary(model):
		print("=" * 80)
		print("HYBRID IDS MODEL ARCHITECTURE")
		print("=" * 80)
		print(f"Autoencoder Branch: {input_dim_ae} → 64 → 32 → 64 → {input_dim_ae}")
		print(f"CNN Branch: 2x (Conv1D(32) → BatchNorm → MaxPool)")
		print(f"LSTM Branch: LSTM(128) with LayerNorm")
		print(f"Fusion Layer: Combined features ({32+32+128} dimensions)")
		print("=" * 80)
		model.summary()

		
if __name__ == "__main__":
    model = IDSModelFactory.create_model() 
    model.summary()