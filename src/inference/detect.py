import numpy as np
import pandas as pd
from keras.models import load_model
from scapy.all import sniff
from ..utils.logger import IDSLogger
from ..preprocessing import clean, features, normalize

class IDSDetector:

    def __init__(self, model_path='./models/ids.h5', threshold=0.9):
        self.logger = IDSLogger()
        self.model = load_model(model_path)
        self.threshold = threshold
        self.preprocessor = self._init_preprocessor()
        self.logger.log('INFO', 'IDS detector initialized')

    def _init_preprocessor(self):
        return {
            'clean': clean.DataCleaner(),
            'features': features.FeatureHarmonizer(),
            'normalize': normalize.DataNormalizer()
        }

    def process_packet(self, packet):
        try:
            packet_df = self._packet_to_df(packet)
            
            cleaned = self.preprocessor['clean'].transform(packet_df)
            featurized = self.preprocessor['features'].transform(cleaned)
            normalized = self.preprocessor['normalize'].transform(featurized)
            
            prediction = self.model.predict(normalized)
            if prediction > self.threshold:
                self._raise_alert(packet, prediction)
        except Exception as e:
            self.logger.log('ERROR', f'Packet processing failed: {str(e)}')

    def _packet_to_df(self, packet):
        return pd.DataFrame([{
            'src_ip': packet['IP'].src,
            'dst_ip': packet['IP'].dst,
            'protocol': packet['IP'].proto,
            'length': len(packet),
            'timestamp': packet.time,
        }])

    def _raise_alert(self, packet, score):
        alert_msg = (f"ALERT: Malicious traffic detected from {packet['IP'].src} "
                    f"to {packet['IP'].dst} (score: {score[0][0]:.2f})")
        self.logger.log('WARNING', alert_msg)

    def start_monitoring(self, interface='eth0'):
        """Start live packet monitoring"""
        self.logger.log('INFO', f'Starting monitoring on interface {interface}')
        sniff(iface=interface, prn=self.process_packet, store=False)

if __name__ == "__main__":
    detector = IDSDetector()
    detector.start_monitoring()