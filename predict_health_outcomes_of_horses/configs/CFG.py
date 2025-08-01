import os

class CFG:

    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    BASE_PATH = os.path.join(ROOT_DIR, "data")
    
    TARGET = 'outcome'
    
    TRAIN = True
    n_splits = 5
    test_size = 0.3
    random_state = 42
    verbose = False

    n_classes = 3

    nan_cols = ['temp_of_extremities', 'peripheral_pulse',
                 'capillary_refill_time','pain', 'peristalsis','abdominal_distention',
                 'nasogastric_tube','nasogastric_reflux','rectal_exam_feces',
                 'abdomen','abdomo_appearance']
    
    binary_col = ["surgery", "age", "surgical_lesion", "cp_data"]
    
    ohe_cols = ["mucous_membrane"]

    # base model = 
    decision_function_shape = 'ovo'

    n_estimators = 10000
    device = 'cpu'

    # optium 
    n_trials = 10000
    early_stopping_rounds = 100

    class_weights_dict = {0:0.7, 1:1, 2:0.45}

    debug = False

    def update_debug(self):
        if self.debug:
            self.n_trials = 10
            self.n_estimators = 5
            self.n_splits = 2
            self.early_stopping_rounds = 5