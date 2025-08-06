from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.regularizers import l2

# import mlflow
# mlflow.tensorflow.log_model(model, "model")

# from multiprocessing import Pool
# pool = Pool(processes=4)

# import smtplib
# def send_email_alert(alert_msg):

class IDSModelFactory:
	@staticmethod
	def create_mlp(input_dim, num_classes=1):
		model = Sequential([
			Dense(64, activation='relu', input_shape=(input_dim,), kernel_regularizer=l2(0.01)),
			Dropout(0.3),
			Dense(32, activation='relu'),
			Dense(num_classes, activation='sigmoid')
		])
		return model

	@staticmethod
	def create_lstm(input_shape, num_classes=1):
		model = Sequential([
			LSTM(64, return_sequences=True, input_shape=input_shape),
			Dropout(0.2),
			LSTM(32),
			Dense(num_classes, activation='sigmoid')
		])
		return model
	
	@staticmethod
	def create_cnn(input_shape, num_classes=1):
		model = Sequential([
			Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
			MaxPooling1D(pool_size=2),
			Conv1D(32, kernel_size=3, activation='relu'),
			Flatten(),
			Dense(num_classes, activation='sigmoid')
		])
		return model

	@staticmethod
	def create_autoencoder(input_dim, encoding_dim=8):
		input_layer =  Input(shape=(input_dim,))
		encoder = Dense(encoding_dim, activation='relu')(input_layer)
		decoder = Dense(input_dim, activation='sigmoid')(encoder)
		model = Model(inputs=input_layer, outputs=decoder)
		return model
		
if __name__ == "__main__":
    model = IDSModelFactory.create_lstm(input_shape=(10, 15))  # 10 timesteps, 15 features
    model.summary()