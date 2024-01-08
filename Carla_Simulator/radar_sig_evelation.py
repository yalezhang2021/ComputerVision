


import numpy as np
import matplotlib.pyplot as plt
import h5py

class RadarSimSignalData:
   def __init__(self):
      self.tx_positions = np.empty(1)
      self.rx_positions = np.empty(1)
      self.number_chirps = 0
      self.bandwidth = 0.0
      self.chirp_duration = 0.0
      self.cycle_duration = 0.0
      self.signals = np.empty(1)
      self.time_vector = np.empty(1)
      self.description = ""
      self.carrier_frequency = 0.0


def load_radar_sim_signal_data_h5(filename):

   radar_sim_signal = RadarSimSignalData()
   hf = h5py.File(filename, 'r')
   radar_sim_signal.tx_positions = np.array(hf.get("tx_positions"))
   radar_sim_signal.rx_positions = np.array(hf.get("rx_positions"))
   radar_sim_signal.bandwidth = np.array(hf.get("bandwidth"))
   radar_sim_signal.chirp_duration = np.array(hf.get("chirp_duration"))
   radar_sim_signal.cycle_duration = np.array(hf.get("cycle_duration"))
   radar_sim_signal.signals = np.array(hf.get("signals"))
   radar_sim_signal.time_vector = np.ravel(hf.get("time_vector"))
   radar_sim_signal.carrier_frequency = np.array(hf.get("carrier_frequency"))

   return radar_sim_signal

def create_radar_image(radar_sim_data):

    # python: index 0: tx-antenna, index 1: rx-antenna, index 2: chirp/doppler, index 3: time signal/range
    # matlab: index 1: tx-antenna, index 2: rx-antenna, index 3: chirp/doppler, index 4: time signal/range

    if_signal = radar_sim_data.signals
    print(radar_sim_data.description)
    number_tx = if_signal.shape[0]
    number_rx = if_signal.shape[1]
    number_chirps = if_signal.shape[2]
    number_samples = if_signal.shape[3]
    number_virt_antennas = number_tx*number_rx
    virt_signal = np.zeros((number_tx*number_rx, number_chirps, number_samples), dtype=np.complex128)

    for tx_index in range(number_tx):
        for rx_index in range(number_rx):
            virt_index = tx_index*number_rx + rx_index
            virt_signal[virt_index] = if_signal[tx_index, rx_index]


    # zero padding and windowing
    zero_padded_virt_signal = np.zeros((128, 128, 512), dtype=np.complex128)

    hanning_single_angle = np.hanning(number_virt_antennas+2)[1:-1, np.newaxis]
    hanning_single_doppler = np.hanning(number_chirps+2)[1:-1, np.newaxis]
    hanning_single_range = np.hanning(number_samples+2)[1:-1, np.newaxis]
    hanning_window = hanning_single_angle@hanning_single_doppler.T
    hanning_window = hanning_window[..., np.newaxis]
    hanning_window = hanning_window@hanning_single_range.T

    zero_padded_virt_signal[:virt_signal.shape[0], :virt_signal.shape[1], :virt_signal.shape[2]] = hanning_window*virt_signal

    # create 3D-fft angle, doppler, range
    reco_3d = np.fft.fftn(zero_padded_virt_signal)
    reco_3d = np.fft.fftshift(reco_3d, axes=(0,1))
    reco_3d = np.abs(reco_3d)

    # create slice for angle and range
    reco_angle_range = np.max(reco_3d, axis=1)
    max_value = np.max(reco_angle_range)
    plt.figure("reco_angle_range")
    plt.imshow(reco_angle_range, aspect="auto", vmax=max_value/2.0)
    plt.savefig("reco_angle_range.png", bbox_inches="tight")
    plt.close()

    # create slice for doppler and range
    reco_doppler_range = np.max(reco_3d, axis=0)
    max_value = np.max(reco_doppler_range)

    plt.figure("reco_doppler_range")
    plt.imshow(reco_doppler_range, aspect="auto", vmax=max_value/2.0)
    plt.savefig("reco_doppler_range.png", bbox_inches="tight")
    plt.close()

    # generate detections
    max_value_of_reco = np.max(reco_3d)
    threshold = max_value_of_reco*0.8  # or any suitable value
    indices = np.argwhere(reco_3d > threshold)

    indices_angle_range = indices[:, (0,2)]
    
    plt.figure("scatter_detections")
    plt.scatter(indices_angle_range[:, 1], indices_angle_range[:,0])
    plt.xlim(0, 512)
    plt.ylim(128, 0)

    plt.savefig("scatter_plot.png", bbox_inches='tight')
    plt.close()
    
def main():
    radar_sim_data = load_radar_sim_signal_data_h5("if_signal_0000.h5")
    create_radar_image(radar_sim_data)

if __name__ == "__main__":
    main()