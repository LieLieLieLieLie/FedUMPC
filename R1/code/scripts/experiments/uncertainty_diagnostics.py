"""Generate held-out uncertainty records from the primary predictors."""
from config import Config
import r1_experiments as exp

if __name__ == '__main__':
    Config.set_mode('fast')
    _, metadata, prox, *_ = exp.prepare_primary_models()
    exp.uncertainty_diagnostics(prox, metadata)
