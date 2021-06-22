
load('VNM.mat'); % load the VNM,
load('net.mat'); %load the net


%READ A DISTORTED IMAGE 
Id = imread('1-1.png');
B1=CSIQVNM(Id, VNM);
Score  = sim(net,B1);











