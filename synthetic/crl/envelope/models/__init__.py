from __future__ import absolute_import, division, print_function
from .linear import EnvelopeLinearCQN
from .linear import Actor,Critic


def get_new_model(name, state_size, action_size, reward_size):
    if name == 'linear':
        m = EnvelopeLinearCQN(state_size, action_size, reward_size)
        a = Actor(state_size, action_size, reward_size)
        a_t = Actor(state_size, action_size, reward_size)
        c = Critic(state_size, action_size, reward_size)
        c_t = Critic(state_size, action_size, reward_size)
        return m,a,a_t,c,c_t
    else:
        print("model %s doesn't exist." % (name))
        return None
