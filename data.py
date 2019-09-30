import numpy as np

class DataGenerator:
    def __init__(self, datasource, data_length=500):
        self.data_length = 500
        self.datasource = datasource
    
    def create_sine_wave(self, n, steps_in=100):
        n += steps_in
        ix = np.arange(n) + 1
        f = 5#np.random.uniform(size=n, low=5, high=0.499)
        A = np.random.uniform(low=0.9, high=0.89999, size=n)
        offset = 0#np.random.uniform(low=-np.pi, high=np.pi, size=n)
        sine_wave = A*np.sin(2*np.pi*f*ix/float(n) + offset)
        df_seq = Sequence(sine_wave)
        
        reframed = df_seq.split(steps_in=steps_in, steps_out=0, feature_names=['V'])
        values = reframed.values
        return values.reshape(values.shape[0], 1, steps_in)