function B = CSIQVNM(Id, VNM)

patchSize = sqrt(size(VNM,2)/3);
patchDim = size(VNM,2);

%%%%%% DIVIDING EACH IMAGE INTO BLOCKS %%%%%%
startPosition = 1;
%Id=rgb2gray(Id);%grayimage
sizeY = size(Id,1); sizeX = size(Id,2);%
gridY = startPosition : patchSize : sizeY-patchSize; % 
gridX = startPosition : patchSize : sizeX-patchSize; %

Y = length(gridY);  X = length(gridX);%

Xd = zeros(patchDim, Y*X);
Xd1 = zeros(256, Y*X);

ij = 0;
ij1=0;
%------------ªÒ»°Xd--------------%
for i = gridY
    for j = gridX
        ij = ij+1; 
        Xd(:,ij) = reshape( Id(i:i+patchSize-1, j:j+patchSize-1,1:3), [patchDim 1] );
    end
end

Xd = double(Xd);   
mXd = mean(Xd);     
varXd=var(Xd);
meXd = mean(mXd); 
% remove mean value
Xd = (Xd-ones(size(Xd,1),1)*meXd);

B2 = VNM*Xd;
B=std(B2,0,2);%standard





