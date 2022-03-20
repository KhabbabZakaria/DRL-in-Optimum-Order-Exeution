# DRL-in-Optimum-Order-Exeution

Inputs: 2 types of inputs are used. 1) Public Input which is the Market States for the stocks. At a time, 10 previous minutes of Market Information is nfed.
                                    2) Private Input which is the Left Time and Left Executed Order
                                    

Model: The 2 types of inputs are fed into 2 RNN Networks. Their outputs are concatenated and fed into an Actor Critic Netwrok. The Output of the Actor is the fraction of volume to be traded at the next minute. 

![DRL](https://user-images.githubusercontent.com/46716277/159191098-c9407b01-edf7-4df5-960b-e5fa177470cd.png)

