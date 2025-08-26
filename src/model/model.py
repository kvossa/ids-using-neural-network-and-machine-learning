from keras.models import Model, Sequential
from keras.layers import Input, Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten, concatenate
from keras.regularizers import l2

# import mlflow
# mlflow.tensorflow.log_model(model, "model")

# from multiprocessing import Pool
# pool = Pool(processes=4)

# import smtplib
# def send_email_alert(alert_msg):

class IDSModelFactory:
	@staticmethod
	def create_model(input_dim_lstm=(100,10), input_dim_cnn=(100,10), input_dim_ae=20, num_classes=1):
		lstm_input = Input(shape = input_dim_lstm, name="lstm_input")
		lstm_out = LSTM(64, return_sequences = True)(lstm_input)
		lstm_out = Dropout(0.2)(lstm_out)
		lstm_out = LSTM(32)(lstm_out)

		cnn_input = Input(shape=input_dim_cnn, name = "cnn_input")
		cnn_out = Conv1D(64, kernel_size=3, activation='relu')(cnn_input)
		cnn_out = MaxPooling1D(pool_size=2)(cnn_out)
		cnn_out = Conv1D(32, kernel_size=3, activation='relu')(cnn_out)
		cnn_out = Flatten()(cnn_out)
		
		ae_input = Input(shape=(input_dim_ae), name='ae_input')
		encoded = Dense(32, activation='relu', name='encoded_features')(ae_input)

		combined =  concatenate([lstm_out, cnn_out, encoded])

		dense_out = Dense(64, activation='relu')(combined)
		output = Dense(num_classes, activation='sigmoid', name='classification_output')(dense_out)

		model = Model(
			inputs=[lstm_input, cnn_input, ae_input],
			outputs=output,
		)
		return model
		
if __name__ == "__main__":
    model = IDSModelFactory.create_model() 
    model.summary()