from __future__ import absolute_import, division, print_function
import random
import torch
import copy
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from collections import namedtuple
from collections import deque

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


class MetaAgent(object):
    '''
    (1) act: how to sample an action to examine the learning
        outcomes or explore the environment;
    (2) memorize: how to store observed observations in order to
        help learing or establishing the empirical model of the
        enviroment;
    (3) learn: how the agent learns from the observations via
        explicitor implicit inference, how to optimize the policy
        model.
    '''

    def __init__(self, model, a_model,a_model_t, c_model,c_model_t, args, is_train=False):


        self.actor = a_model
        self.actor_target = copy.deepcopy(a_model)


        self.actor_target.load_state_dict(self.actor.state_dict())

        if args.optimizer == 'Adam':
            self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=args.lr)
        elif args.optimizer == 'RMSprop':
            self.actor_optimizer  = optim.RMSprop(self.actor.parameters(), lr=args.lr)

        self.critic = c_model
        self.critic_target = copy.deepcopy(c_model)

        self.critic_target.load_state_dict(self.critic.state_dict())

        if args.optimizer == 'Adam':
            self.critic_optimizer  = optim.Adam(self.critic.parameters(), lr=args.lr)
        elif args.optimizer == 'RMSprop':
            self.critic_optimizer  = optim.RMSprop(self.critic.parameters(), lr=args.lr)  

        self.model_ = model
        self.model = copy.deepcopy(model)
        self.is_train = is_train
        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_delta = (args.epsilon - 0.05) / args.episode_num

        self.mem_size = args.mem_size
        self.batch_size = args.batch_size
        self.weight_num = args.weight_num

        self.noise            = args.noise
        self.beta            = args.beta
        self.beta_init       = args.beta
        self.homotopy        = args.homotopy
        self.beta_uplim      = 1.00
        self.tau             = 1000.
        self.beta_expbase    = float(np.power(self.tau*(self.beta_uplim-self.beta), 1./args.episode_num))
        self.beta_delta      = self.beta_expbase / self.tau

        self.trans_mem = deque()
        self.trans = namedtuple('trans', ['s', 'a', 's_', 'r', 'd'])
        self.priority_mem = deque()

        if args.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model_.parameters(), lr=args.lr)
        elif args.optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(self.model_.parameters(), lr=args.lr)

        self.w_kept = None
        self.update_count = 0
        self.update_freq = args.update_freq

        if self.is_train:
            self.model.train()
        if use_cuda:
            self.model.cuda()
            self.model_.cuda()

    def act(self, state, preference=None):
        # random pick a preference if it is not specified
        if preference is None:
            if self.w_kept is None:
                self.w_kept = torch.randn(self.critic.reward_size)
                self.w_kept = (torch.abs(self.w_kept) / \
                               torch.norm(self.w_kept, p=1)).type(FloatTensor)
            preference = self.w_kept
        state = torch.from_numpy(state).type(FloatTensor)

        action = self.actor(
            Variable(state.unsqueeze(0)),
            Variable(preference.unsqueeze(0)),
            Variable(preference.unsqueeze(0)))


        action += (1**0.5)*torch.randn(action.shape).to('cuda')
    
        return action.cpu()

    def memorize(self, state, action, next_state, reward, terminal):
       
        self.trans_mem.append(self.trans(
            torch.from_numpy(state).type(FloatTensor),  # state
            action,  # action
            torch.from_numpy(next_state).type(FloatTensor),  # next state
            torch.from_numpy(reward).type(FloatTensor),  # reward
            terminal))  # terminal

        # randomly produce a preference for calculating priority
        # preference = self.w_kept
        preference = torch.randn(self.critic.reward_size)
        preference = (torch.abs(preference) / torch.norm(preference, p=1)).type(FloatTensor)
        state = torch.from_numpy(state).type(FloatTensor)

        q = self.critic(Variable(state.unsqueeze(0), requires_grad=False),
                            Variable(action.unsqueeze(0), requires_grad=False),
                           Variable(preference.unsqueeze(0), requires_grad=False)).squeeze()

        q = q.data
        wq = preference.dot(q)

        wr = preference.dot(torch.from_numpy(reward).type(FloatTensor))
        if not terminal:
            next_state = torch.from_numpy(next_state).type(FloatTensor)
            next_action = self.actor(Variable(next_state.unsqueeze(0), requires_grad=False),
                                Variable(preference.unsqueeze(0), requires_grad=False),
                                Variable(preference.unsqueeze(0), requires_grad=False))
            hq = self.critic(Variable(next_state.unsqueeze(0), requires_grad=False),
                            Variable(next_action.unsqueeze(0), requires_grad=False),
                           Variable(preference.unsqueeze(0), requires_grad=False))

            hq = hq.data[0]
            
            whq = preference.dot(hq)
            p = abs(wr + self.gamma * whq - wq)
        else:
            print(self.beta)
            self.w_kept = None
            if self.epsilon_decay:
                self.epsilon -= self.epsilon_delta
            if self.homotopy:
                self.beta += self.beta_delta
                self.beta_delta = (self.beta-self.beta_init)*self.beta_expbase+self.beta_init-self.beta
            p = abs(wr - wq)
        p += 1e-5

        self.priority_mem.append(
            p
        )
        if len(self.trans_mem) > self.mem_size:
            self.trans_mem.popleft()
            self.priority_mem.popleft()

    def sample(self, pop, pri, k):
        a = []
        for i in pri:
            a.append(i.cpu())
        pri = np.array(a).astype(np.float)
        inds = np.random.choice(
            range(len(pop)), k,
            replace=False,
            p=pri / pri.sum()
        )
        return [pop[i] for i in inds]

    def actmsk(self, num_dim, index):
        mask = ByteTensor(num_dim).zero_()
        mask[index] = 1
        return mask.unsqueeze(0)

    def nontmlinds(self, terminal_batch):
        mask = ByteTensor(terminal_batch)
        inds = torch.arange(0, len(terminal_batch)).type(LongTensor)
        inds = inds[mask.eq(0)]
        return inds

    def learn(self):
        if len(self.trans_mem) > self.batch_size:

            self.update_count += 1

            action_size = self.model_.action_size
            reward_size = self.model_.reward_size

            minibatch = self.sample(self.trans_mem, self.priority_mem, self.batch_size)
            # minibatch = random.sample(self.trans_mem, self.batch_size)
            batchify = lambda x: list(x)
            state_batc = batchify(map(lambda x: x.s.unsqueeze(0), minibatch))
            action_batc = batchify(map(lambda x: x.a.unsqueeze(0), minibatch))
            reward_batc = batchify(map(lambda x: x.r.unsqueeze(0), minibatch))
            next_state_batc = batchify(map(lambda x: x.s_.unsqueeze(0), minibatch))
            terminal_batc = batchify(map(lambda x: x.d, minibatch))


            w_batch = np.random.randn(self.weight_num, reward_size)
            w_batch = np.abs(w_batch) / \
                      np.linalg.norm(w_batch, ord=1, axis=1, keepdims=True)
            w_ = w_batch.repeat(self.weight_num, axis=0)
            w_i = w_
            for i in range(self.batch_size-1):
                w_i = np.concatenate((w_i,w_))
            w_i = torch.from_numpy(w_i).type(FloatTensor)
            w_ii = w_batch
            for i in range(self.weight_num-1):
                w_ii = np.concatenate((w_ii,w_batch))
            w_iii = w_ii
            for i in range(self.batch_size-1):
                w_iii = np.concatenate((w_iii,w_ii))
            w_ii = torch.from_numpy(w_iii).type(FloatTensor)

            
            next_state = torch.cat(next_state_batc, dim=0)
            next_state = next_state.repeat_interleave(self.weight_num*self.weight_num, dim=0)

            next_action = self.actor_target(Variable(next_state, requires_grad=False),
                                            Variable(w_i, requires_grad=False),
                                            Variable(w_ii, requires_grad=False))
            
            # next_action = (next_action == next_action.max(dim=1, keepdim=True)[0]).to(dtype=torch.int32)

            Q_ = self.critic_target(Variable(next_state, requires_grad=False),
                                            Variable(next_action, requires_grad=False),
                                            Variable(w_ii, requires_grad=False))

            Q__ = Q_.view(-1,self.weight_num,self.critic.reward_size)
            sum_Q_ = torch.bmm(Variable(w_i.unsqueeze(1), requires_grad=False),Q_.unsqueeze(2)).squeeze() \
                    .view(-1,self.weight_num)
            _, indices_max = sum_Q_.max(dim=1, keepdim=True)

            max_0 = indices_max[0]
            max_45 = indices_max[45]
            a = Q__[0][max_0].squeeze()
            aa = Q__[45][max_45].squeeze()
            n = torch.arange(0,self.batch_size*self.weight_num).to('cuda')
            n = torch.cat((n.unsqueeze(1),indices_max),1)
            HQ = torch.zeros(self.batch_size*self.weight_num,self.critic.reward_size, dtype=torch.float32).to('cuda')
            for i in range(self.batch_size*self.weight_num):
                HQ[i] = Q__[i][indices_max[i]]

            # HQ = Q__.gather(1,indices_max.squeeze())



            state = torch.cat(state_batc, dim=0)
            state = state.repeat_interleave(self.weight_num, dim=0)

            action = torch.cat(action_batc, dim=0)
            action = action.repeat_interleave(self.weight_num, dim=0).squeeze()
            w_iii = w_batch
            for i in range(self.batch_size-1):
                w_batch = np.concatenate((w_batch,w_iii))
            w_batch = torch.from_numpy(w_batch).type(FloatTensor)

            Q = self.critic(Variable(state),
                            Variable(action),
                            Variable(w_batch))
            
            terminal = np.repeat(terminal_batc, self.weight_num)
            

            nontmlmask = self.nontmlinds(terminal)
            with torch.no_grad():
                Tau_Q = Variable(torch.zeros(self.batch_size * self.weight_num,
                                             reward_size).type(FloatTensor)) 
                Tau_Q[nontmlmask] = self.gamma * HQ[nontmlmask]
                # Tau_Q.volatile = False
                reward = torch.cat(reward_batc, dim=0)
                reward = reward.repeat_interleave(self.weight_num, dim=0).squeeze()
                Tau_Q += Variable(reward)

            
            Tau_Q = Tau_Q.view(-1, reward_size)

            wQ = torch.bmm(Variable(w_batch.unsqueeze(1)),
                           Q.unsqueeze(2)).squeeze()

            wTQ = torch.bmm(Variable(w_batch.unsqueeze(1)),
                            Tau_Q.unsqueeze(2)).squeeze()
            

            act = self.actor(Variable(state),
                            Variable(w_batch),
                            Variable(w_batch))


            Q_act = self.critic(Variable(state),
                act,
                Variable(w_batch))

            act_loss= -(torch.bmm(Variable(w_batch.unsqueeze(1), requires_grad=True),
                           Q_act.unsqueeze(2)).squeeze()).mean()
            
            # act_loss = (1-self.beta) * (-act_loss.mean())

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            act_loss.backward()
            for param in self.actor.parameters():
                param.grad.data.clamp_(-1, 1)
            self.actor_optimizer.step()

            # loss = F.mse_loss(Q.view(-1), Tau_Q.view(-1))
            cri_loss = self.beta * F.mse_loss(wQ.view(-1), wTQ.view(-1))
            cri_loss += (1-self.beta) * F.mse_loss(Q.view(-1), Tau_Q.view(-1))
            
            self.critic_optimizer.zero_grad()
            cri_loss.backward()
            for param in self.critic.parameters():
                param.grad.data.clamp_(-1, 1)
            self.critic_optimizer.step()

            # act_loss = -self.critic(Variable(state),
            #                 Variable(action),
            #                 Variable(w_batch)).mean()

            
            if self.update_count % self.update_freq == 0:
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(0.005 * param.data + (1 - 0.005) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(0.005 * param.data + (1 - 0.005) * target_param.data)

            return cri_loss.data, act_loss.data

        return 0.0, 0.0
            

    def reset(self):
        self.w_kept = None
        if self.noise > 0:
            self.noise -= (0.01)/1000
        else:
            print(1)
        if self.epsilon_decay:
            self.epsilon -= self.epsilon_delta
        if self.homotopy:
            self.beta += self.beta_delta
            self.beta_delta = (self.beta-self.beta_init)*self.beta_expbase+self.beta_init-self.beta

    def predict(self,state,action, probe):
        return self.critic(Variable(FloatTensor(state).unsqueeze(0), requires_grad=False),
                        Variable(FloatTensor(action).unsqueeze(0), requires_grad=False),
                          Variable(probe.unsqueeze(0), requires_grad=False))

    def save(self, save_path, model_name):
        torch.save(self.model, "{}{}.pkl".format(save_path, model_name))


    def find_preference(
            self,
            w_batch,
            target_batch,
            pref_param):

        with torch.no_grad():
            w_batch = FloatTensor(w_batch)
            target_batch = FloatTensor(target_batch)

        # compute loss
        pref_param = FloatTensor(pref_param)
        pref_param.requires_grad = True
        sigmas = FloatTensor([0.001]*len(pref_param))
        dist = torch.distributions.normal.Normal(pref_param, sigmas)
        pref_loss = dist.log_prob(w_batch).sum(dim=1) * target_batch

        self.optimizer.zero_grad()
        # Total loss
        loss = pref_loss.mean()
        loss.backward()
        
        eta = 1e-3
        pref_param = pref_param + eta * pref_param.grad
        pref_param = simplex_proj(pref_param.detach().cpu().numpy())
        # print("update prefreence parameters to", pref_param)

        return pref_param


# projection to simplex
def simplex_proj(x):
    y = -np.sort(-x)
    sum = 0
    ind = []
    for j in range(len(x)):
        sum = sum + y[j]
        if y[j] + (1 - sum) / (j + 1) > 0:
            ind.append(j)
        else:
            ind.append(0)
    rho = np.argmax(ind)
    delta = (1 - (y[:rho+1]).sum())/(rho+1)
    return np.clip(x + delta, 0, 1)
