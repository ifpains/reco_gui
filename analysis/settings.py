

class FilterSettings:
    # constructor
    def __init__(self):
        self.data_folder = '../simulations.json'
        self.noise_file = '../pedestals/pedmap_run02054_rebin1.root'
        #self.packs = ['Digi_He_60_kev', 'Digi_CYGNO_60_40_ER_60_keV_part39']
        self.packs = ['../histograms/histograms_Run02163.root']
        self.preprocessing_params = ('../new_threshold_params_cubic.json', ['Digi_CYGNO_60_40_ER_60_keV_part39', 'Digi_He_60_kev'], ['median', 'cygno', 'gaussian', 'mean', 'unet'])
        self.run_number = 817
        self.sup = 16
        self.inf = -26
        self.roc_grid = 120
        self.nsamples = 100
        self.output_file_path = '../'
        self.output_file_name = 'threshold_changes_v3_er'
        self.filters = {

                        #'bilateral': [[i, j, k] for i in range(3, 19, 2) for j in range(1, 5, 2) for k in range(1, 19, 2)],
                        #'nlmeans': [[i, j] for i in range(1, 25, 3) for j in range(1, 25, 3)],
                        #'nlmeans':[[11, 7]],
                        'unet': [[0]],
                        'mean': [[17]],
                        'gaussian': [[17]],
                        'median': [[15]],
                        #'wiener': [[1], [2]],
                        #'bm3D': [[1], [2], [3], [4], [5], [6], [7], [8]],
                        'cygno': []
                        #'tv': [[.1], [.2], [.3], [.4], [.5], [.6], [.7], [.8], [.9]],
                        #'wavelets' : [[None]]
                        }

