load('VNM.mat'); % load the VNM,
load('net.mat'); %load the net


%READ A DISTORTED IMAGE 
Id = imread('1-1.png');
B=VNM_NN(Id, VNM);
Score  = sim(net,B)











