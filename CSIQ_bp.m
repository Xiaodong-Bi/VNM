PLCC=zeros(6,100);
SRCC=zeros(6,100);

load('CSIQ.mat');   % load database information and DMOS data  
load('VNM.mat');     % load the feature detector, Visual Neuron Matrix VNM

B=[];
BB=[];
Mos=[];
disnum=6;
N=866

%Extract the features of the image
for iPoint = 1:866
    %READ A DISTORTED IMAGE  
    Id = imread(['F:\ImageDate\CSIQ\dst_imgs\' csiq_imTitle{iPoint,1} '\' csiq_imTitle{iPoint,2} '.' csiq_imTitle{iPoint,1} '.' num2str(csiq_imDMOS(iPoint,1)) '.png']);
    
    B1=VNM_NN(Id, VNM);
    B=[B B1];
    MOS1=csiq_imDMOS(iPoint,2);
    B_row=size(B1,1);
    
    Mos1=linspace(MOS1,MOS1,B_row);
    Mos1=Mos1';
    
    Mos=[Mos MOS1];
end

%TRAINING
for SS=1:5
    
    Trainnum=700;
    T=10;
    randIndex = randperm(size(B,2));
    B=B(:,randIndex);
    Mos=Mos(randIndex);
    
    Ltag=tag(randIndex);
    
    train_data=B(:,1:Trainnum);
    train_value=Mos(1:Trainnum);
    test_data=B(:,1:Trainnum);
    
    %test_data=B(:,Trainnum+1:N);%
    
    Ltag=tag(randIndex);
    
    
    % T=10;
    % train_data=B;
    % train_value=Mos;
    % test_data=B;
    K=size(train_data, 2);% K in paper
    
    
    %initialize the distribution D1 of the training set
    D(1,:)=ones(1,K)/K;
    
    
    for i=1:T %for each WeakLearn
        
        %train ith WeakLearn
        net=newff(train_data, train_value, [8, 8], {'tansig', 'radbas', 'tansig'});
        net.trainParam.epochs=5000;
        net.trainParam.showWindow=0;
        net.trainParam.lr=0.1;
        net=train(net,train_data,train_value);
        
        % estimate the predicted output of the training set
        BPoutput=sim(net,train_data);
        
        %compute the difference between the predicted output of the training
        %set and original ones
        train_error(i,:)=train_value-BPoutput;
        
        % estimate the predicted output of the test set
        test_output(i,:)=sim(net,test_data);
        
        %updata the distribution Di+1 for next WeakLearn and compute the evaluation error of the ith WeakLearn
        all_Error(i)=0;
        for j=1:K
            if abs(train_error(i,j))>0.1 % threshold=0.1
                all_Error(i)=all_Error(i)+D(i,j);
                D(i+1,j)=D(i,j)*1.1;% sigma=0.1
            else
                D(i+1,j)=D(i,j);
            end
        end
        
        %compute the ith WeakLearn weight
        %     a(i)=0.5*log((1+all_Error(i))/(1-all_Error(i)));
        a(i)=sigmf(abs(1-all_Error(i)),[1 0.5]);
        D(i+1,:)=D(i+1,:)/sum(D(i+1,:));
    end
    
    a=a/sum(a);
    
    %the final output
    dec=a*test_output;
    dec=dec';
    

    OB=dec;
    SB=Mos(1:Trainnum);

    PLCC(disnum+1,SS)=corr(SB',OB,'type','pearson') ;
    SRCC(disnum+1,SS)=corr(SB',OB,'type','spearman');
    
    metric1= corr(SB', OB, 'type', 'pearson');
    metric2 = corr(SB', OB, 'type', 'spearman');
    metric3 = corr(SB', OB, 'type', 'kendall');
    figure,scatter(SB', OB,'*');
    c=zeros(1,disnum+1);
    score=zeros(disnum+1,100);
    Tdmos=zeros(disnum+1,100);

    for a=1:Trainnum
        b=a+Trainnum;
        c(Ltag(b))=c(Ltag(b))+1;
        score(Ltag(b),c(Ltag(b)))=OB(a);
        Tdmos(Ltag(b),c(Ltag(b)))=Mos(b);
    end
    

    for a=1:disnum
        
        PLCC(a,SS)=corr(Tdmos(a,1:c(a))',score(a,1:c(a))','type','pearson') ;
        SRCC(a,SS)=corr(Tdmos(a,1:c(a))',score(a,1:c(a))','type','spearman');
        figure,scatter(score(a,1:c(a))',Tdmos(a,1:c(a))','*');
    end
end


