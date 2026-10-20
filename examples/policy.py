import sys
from pathlib import Path
import numpy as np
 
sys.path.insert(0, str(Path(__file__).parent))  # lets you import your own modules
 
 
def load_policy(checkpoint, observation_space, action_space):
    model = load_your_model(checkpoint)
 
    class Policy:
        def reset(self):  # start of each episode
            reset_your_model_state(model)
 
        def __call__(self, obs):  # every 0.05 s of simulation time
            action = predict(model, obs)
            return np.clip(np.asarray(action, dtype=np.float32).reshape(2, 9), -1, 1)
 
    return Policy()