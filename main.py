import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms

import os
import math
import numpy as np
from parameters import *
from rnn import *
from actorcritic import *
from data import *
import torch.nn.functional as F
import torch


"""-----------------------Arguments-----------------------"""
parser = argparse.ArgumentParser(description='Training of UNet2d-Segmentation')
parser.add_argument("--maximum_epochs", default=100, type=int)
parser.add_argument("--lr", default=np.float32(5e-5), type=float)
parser.add_argument("--lr_decay", default=np.float32(1e-4), type=float)
parser.add_argument('--optimizer_choice', default='adam', type=str)
parser.add_argument('--reduce_dataset', default='None', type=str)
parser.add_argument('--seed', default=1, type=int)


"""--------------Models, Hyperparameters, Metrics and Variables--------------"""
arguments = parser.parse_args()
maximum_epochs = arguments.maximum_epochs
lr = arguments.lr
lr_decay = arguments.lr_decay
optimizer_choice = arguments.optimizer_choice
reduce_dataset = arguments.reduce_dataset
seed = arguments.seed



text_file = open("result.txt", "w")
print('start')
text_file.write('ending_time, epoch, day, policyloss, valueloss, Loss')
text_file.write('\n')
text_file.close()

torch.manual_seed(seed)



os.chdir(os.path.dirname(os.path.abspath(__file__)))


current_dir = os.getcwd()
ref_folder = os.path.join(current_dir, 'preprocessed_dir')
files = [name for name in os.listdir(ref_folder) if name!= 'vwap calculation.py']

def calc_twap(df_csv):
    n = len(df_csv) - 1
    price_sum = 0.0
    for i in range(1, n + 1):
        high_price = df_csv['high'].iloc[i]
        low_price = df_csv['low'].iloc[i]
        close = df_csv['close'].iloc[i]
        price = (high_price + low_price + close) / 3
        price_sum += price

    return price_sum / n


ref_file = files[ref_stock]
df = pd.read_csv(os.path.join(ref_folder, ref_file))
df = df.iloc[T*30:-T*100]
twap_price = []
for i in range(int(len(df)/T)):
    price = calc_twap(df.iloc[i*T:(i+1)*T])
    twap_price.append(price)

twap_price = list(itertools.chain.from_iterable(itertools.repeat(x, T) for x in twap_price))
twap_price = np.array(twap_price).reshape(len(all_train_list), T)
price_k_bar = twap_price




if optimizer_choice == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lr_decay, amsgrad=True)
if optimizer_choice == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)


if reduce_dataset == 'None':
    all_train_list = all_train_list
    all_train_vwap_list = all_train_vwap_list
if reduce_dataset == 'two':
    all_train_list = all_train_list[:2]
    all_train_vwap_list = all_train_vwap_list[:2]


total_reward_list = []
total_PA_list = []

def train():
    #beta = model.beta
    #alpha = model.alpha
    global policy_old
    global beta
    for ep in range(maximum_epochs):
        day = 0
        while day<len(all_train_list)-1:
            P_bar_strategy = 0 #AEP
            text_file = open("result.txt", "a")
            policy_losses = []
            value_losses = []
            Distillation_losses = []
            print('epoch = ', ep, ' day = ', day)
            env.reset()
            state, price_time_plus_1 = getState(all_train_list[day], all_train_vwap_list[day], env.time, state_size)
            for current_time in range(state_size, T-1):
                #left_target = [Q-x for x in env.already_bought_list]
                #left_target_tensor = torch.Tensor(left_target)
                already_bought_tensor = torch.Tensor(env.already_bought_list)
                left_time = T - env.time
                left_time_tensor = torch.Tensor(list(range(left_time+1,left_time+1+state_size)))
                private_input = torch.stack((already_bought_tensor, left_time_tensor)).reshape(state_size, 2)
                private_input = private_input.to(device)
                state = state.to(device)
                action, Value = agent.act(private_input, state)
                action = action.to(device)
                Value = Value.to(device)
                inference_state = model.inference_state
                inference_state = inference_state.to(device)
                reward = (price_time_plus_1/price_bar[day, env.time] - 1)*action - action**2 #R+ - R-
                P_bar_strategy = P_bar_strategy + action*price_time_plus_1


                env.step(action)
                next_state, price_time_plus_2 =  getState(all_train_list[day], all_train_vwap_list[day], env.time, state_size)

                #left_target = [Q-x for x in env.already_bought_list]
                #left_target_tensor = torch.Tensor(left_target)
                already_bought_tensor = torch.Tensor(env.already_bought_list)
                left_time = T - env.time
                left_time_tensor = torch.Tensor(list(range(left_time+1,left_time+1+state_size)))
                private_input = torch.stack((already_bought_tensor, left_time_tensor)).reshape(state_size, 2)
                private_input = private_input.to(device)
                next_state = next_state.to(device)

                _, next_Value = agent.act(private_input, next_state)

                A_hat = reward + gamma*next_Value - Value
                pdf_prob_now = find_pdf(model.policy, action, inference_state)
                pdf_prob_old = find_pdf(policy_old, action, inference_state)

                mu1, sigma1, _ = model.policy(inference_state)
                mu2, sigma2, _ = policy_old(inference_state)
                tup1 = (mu1, sigma1)
                tup2 = (mu2, sigma2)

                kl = KLDiv(tup1, tup2)
                #print(kl)
                d = (pdf_prob_now/pdf_prob_old)*A_hat
                Lp = -(d - beta*kl)
                policy_losses.append(Lp)
                if Lp<0:
                    print(Lp, (pdf_prob_now/pdf_prob_old), A_hat, kl, reward, next_Value, Value)
                Vt = gamma*next_Value.to(device) + reward.to(device)
                #Lv = F.smooth_l1_loss(Vt, Value)
                Lv = Vt - Value
                value_losses.append(Lv)
                #print('Lv', Lv)
                Ld = 0 #need to adjust............................................................................................
                Distillation_losses.append(torch.Tensor(Ld))
                state = next_state
                price_time_plus_1 = price_time_plus_2
                if left_time == 1 and env.already_bought>0:    #no more time left yet target left
                    print('no more time left yet target left')
                    policy_losses.append(torch.Tensor([100]))

                if env.done == True:
                    break
            total_reward_list.append(reward.cpu())
            PA = ((P_bar_strategy/price_k_bar[day,0]) - 1)
            total_PA_list.append(PA.cpu())



            print('ending time', current_time)
            optimizer.zero_grad()
            if day == len(all_train_list):
                with torch.no_grad():
                    policy_old = copy.deepcopy(model.policy)
            print(torch.stack(policy_losses).mean().to(device) , alpha*torch.stack(value_losses).mean().to(device))
            #loss = torch.stack(policy_losses).mean().to(device) + alpha*torch.stack(value_losses).mean().to(device) + mu*torch.stack(Distillation_losses).mean().to(device)
            loss = torch.stack(policy_losses).mean().to(device) + alpha*torch.stack(value_losses).mean().to(device) - PA
            text_file.write(str(current_time) + ',' + str(ep) + ',' + str(day) + ',' + str(torch.stack(policy_losses).mean().item()) + ',' + str(alpha*torch.stack(value_losses).mean().item()) + ',' + str(loss.item()))
            text_file.write('\n')
            text_file.close()
            print('loss',loss.item())
            #writer.add_scalar('Loss/train', loss.cpu().detach().numpy(), day)

            loss.backward()
            optimizer.step()

            day = day + 1

            if torch.stack(policy_losses).mean().to(device) >= 0:
                torch.save(model.state_dict(), 'seed_'+ str(seed) + '_' + PATH)
            else:
                break

        
        if torch.stack(policy_losses).mean().to(device) >= 0:
            torch.save(model.state_dict(), 'seed_'+ str(seed) + '_' + PATH)
        else:
            break



    plot1 = plt.figure(1)
    x = list(range(len(total_reward_list)))
    plt.plot(x, total_reward_list)
    plt.xlabel("(iterations + 1)*days")
    plt.ylabel("total rewards")
    plt.savefig("Total Rewards.png")
    plt.show()

    plot1 = plt.figure(2)
    x = list(range(len(total_PA_list)))
    plt.plot(x, total_PA_list)
    plt.xlabel("(iterations + 1)*days")
    plt.ylabel("total PA")
    plt.savefig("Total PA.png")
    plt.show()


        
if __name__ == "__main__":
    train()



            

        
        
        


        



