"""Train or load the ten primary closed-loop predictor checkpoints."""
import r1_experiments as exp
from config import Config

if __name__=='__main__':
    Config.set_mode('fast')
    exp.prepare_primary_models()
    print('Primary predictors ready:',exp.MODEL_DIR)
