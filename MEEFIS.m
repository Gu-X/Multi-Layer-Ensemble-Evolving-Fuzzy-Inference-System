function [Output]=MEEFIS(Dataset,Setting)
Layer=Setting.Layer;
Epoch=Setting.Epoch;
data0=Dataset.training;
Y0=Dataset.traininglabels;
data1=Dataset.testing;
NumLayer=length(Layer);
NumElement=zeros(1,NumLayer);
Systm={};
for kk=1:1:NumLayer
    NumElement(kk)=size(Layer{kk},1);
end
Systm{NumLayer,max(NumElement)}=[];
Input1{NumLayer,max(NumElement)}=[];
Output1{NumLayer,max(NumElement)}=[];
Input2{NumLayer,max(NumElement)}=[];
Output2{NumLayer,max(NumElement)}=[];
%%
tic
for kk=1:1:Epoch
    for ii=1:1:NumElement(1)
        Input1{1,ii}.datain=data0(:,Layer{1}(ii,:));
        Input1{1,ii}.dataout=Y0;
        [Output1{1,ii},Systm{1,ii}]=ElementALFS(Input1{1,ii},Systm{1,ii},'Learning');
    end
    for jj=2:1:NumLayer
        for ii=1:1:NumElement(jj)
            Input1{jj,ii}.datain=[];
            for tt=1:1:length(Layer{jj}(ii,:))
                Input1{jj,ii}.datain=[Input1{jj,ii}.datain,Output1{jj-1,Layer{jj}(ii,tt)}.Ye];
            end
            Input1{jj,ii}.dataout=Y0;
            [Output1{jj,ii},Systm{jj,ii}]=ElementALFS(Input1{jj,ii},Systm{jj,ii},'Learning');
        end
    end
end
Output.trainingtime=toc;
for ii=1:1:NumElement(1)
    Input2{1,ii}.datain=data1(:,Layer{1}(ii,:));
    [Output2{1,ii},Systm{1,ii}]=ElementALFS(Input2{1,ii},Systm{1,ii},'Testing');
end
tic
for jj=2:1:NumLayer
    for ii=1:1:NumElement(jj)
        Input2{jj,ii}.datain=[];
        for tt=1:1:length(Layer{jj}(ii,:))
            Input2{jj,ii}.datain=[Input2{jj,ii}.datain,Output2{jj-1,Layer{jj}(ii,tt)}.Ye];
        end
        [Output2{jj,ii},Systm{jj,ii}]=ElementALFS(Input2{jj,ii},Systm{jj,ii},'Testing');
    end
end
Output.testingtime=toc;
Output.systemparam=Systm;
Output.Ye=Output2{NumLayer,ii}.Ye;
end
function [Output,SYST]=ElementALFS(Input,SYST,Mode)
threshold=exp(-1/4);
omega=100;
if strcmp(Mode,'Learning')==1
    if isempty(SYST)==1
        [SYST.MN,SYST.A,SYST.CovMat,SYST.GM,SYST.GX,SYST.C,SYST.P,SYST.LX,SYST.S,SYST.SumL,SYST.Ind,SYST.CarD,Output.Ye]=...
            LearningALM(Input.datain,Input.dataout,omega,threshold);
    else
        [SYST.MN,SYST.A,SYST.CovMat,SYST.GM,SYST.GX,SYST.C,SYST.P,SYST.LX,SYST.S,SYST.SumL,SYST.Ind,SYST.CarD,Output.Ye]=...
            UpdatingALM(SYST.MN,SYST.A,SYST.CovMat,SYST.GM,SYST.GX,SYST.C,SYST.P,SYST.LX,SYST.S,SYST.SumL,SYST.Ind,SYST.CarD,Input.datain,Input.dataout,omega,threshold);
    end
end
if strcmp(Mode,'Testing')==1
    [Output.Ye]=TestingALM(Input.datain,SYST.MN,SYST.A,SYST.C,SYST.LX,SYST.S);
end
end
function [ModelNumber,A,C,Global_mean,Global_X,center,prototype,Local_X,Support,sum_lambda,Index,L1,Ye]=UpdatingALM(ModelNumber,A,C,Global_mean,Global_X,center,prototype,Local_X,Support,sum_lambda,Index,L0,data0,Y0,omega,threshold)
CL=size(Y0,2);
[L2,W]=size(data0);
Ye=zeros(L2,CL);
L1=L2+L0;
%%
for ii=L0+1:1:L1
    [ModelNumber,A,C,Global_mean,Global_X,center,prototype,Local_X,Support,sum_lambda,Index,Ye(ii-L0,:)]=...
        SystemUpdating_st2(ModelNumber,A,C,Global_mean,Global_X,center,prototype,Local_X,Support,sum_lambda,Index,data0(ii-L0,:),Y0(ii-L0,:),ii,omega,threshold,CL,W);
end
end
function [ModelNumber,A,C,Global_mean,Global_X,center,prototype,Local_X,Support,sum_lambda,Index,L,Ye]=LearningALM(data0,Y0,omega,threshold)
CL=size(Y0,2);
[L,W]=size(data0);
center=data0(1,:);
prototype=data0(1,:);
Global_mean=data0(1,:);
Global_X=sum(data0(1,:).^2);
Local_X=sum(data0(1,:).^2);
Support=1;
ModelNumber=1;
sum_lambda=1;
Index=1;
A=zeros(CL,W+1,1);
C=eye(W+1)*omega;
Ye=zeros(L,CL);
%%
for ii=2:1:L
    [ModelNumber,A,C,Global_mean,Global_X,center,prototype,Local_X,Support,sum_lambda,Index,Ye(ii,:)]=...
        SystemUpdating(ModelNumber,A,C,Global_mean,Global_X,center,prototype,Local_X,Support,sum_lambda,Index,data0(ii,:),Y0(ii,:),ii,omega,threshold,CL,W);
end
end
function [ModelNumber,A,C,Global_mean,Global_X,center,prototype,Local_X,Support,sum_lambda,Index,Ye]=...
    SystemUpdating(ModelNumber,A,C,Global_mean,Global_X,center,prototype,Local_X,Support,sum_lambda,Index,datain,yin,ii,omega,threshold,CL,W)
Global_mean=Global_mean.*(ii-1)/ii+datain./ii;
Global_X=Global_X.*(ii-1)/ii+sum(datain.^2)/ii;
datadensity=exp(-1*sum((datain-Global_mean).^2)/(Global_X-sum(Global_mean.^2)+0.0000001));
centerdensity=exp(-1*sum((center-repmat(Global_mean,ModelNumber,1)).^2,2)/(Global_X-sum(Global_mean.^2)+0.0000001));
[centerlambda,LocalDensity]=firingstrength(datain,ModelNumber,center,Local_X,Support);
% [centerlambda,seq1]=ActivatingRules(ModelNumber,centerlambda,LocalDensity);
% Ye=OutputGeneration(datain,A(:,:,seq1),centerlambda(seq1),length(seq1),CL);
Ye=OutputGeneration(datain,A,centerlambda,ModelNumber,CL);
seq=find(LocalDensity>threshold);
if isempty(seq)~=1
    OL=1;
else
    OL=0;
end
if (datadensity>max(centerdensity)||datadensity<min(centerdensity)) && OL==0
    %% new cloud add
    ModelNumber=ModelNumber+1;
    center=[center;datain];
    prototype=[prototype;datain];
    Support=[Support,1];
    Local_X=[Local_X,sum(datain.^2)];
    sum_lambda=[sum_lambda;0];
    Index=[Index,ii];
    A(:,:,ModelNumber)=mean(A,3);
    C(:,:,ModelNumber)=eye(W+1)*omega;
else
    %% local_parameters_update
    [~,label0]=min(pdist2(datain,center));
    Support(label0)=Support(label0)+1;
    center(label0,:)=(Support(label0)-1)/Support(label0)*center(label0,:)+datain/Support(label0);
    Local_X(label0)=(Support(label0)-1)/Support(label0)*Local_X(label0)+sum(datain.^2)/Support(label0);
end
forgettingfactor=0;
[centerlambda,LocalDensity]=firingstrength(datain,ModelNumber,center,Local_X,Support);
sum_lambda=sum_lambda+centerlambda;
utility=sum_lambda./(ii-Index)';
seq=find(utility>=forgettingfactor);
ModelNumber0=length(seq);
if ModelNumber0<ModelNumber
    center=center(seq,:);
    Local_X=Local_X(seq);
    Index=Index(seq);
    sum_lambda=sum_lambda(seq);
    centerlambda=centerlambda(seq)./sum(centerlambda(seq));
    Support=Support(seq);
    A=A(:,:,seq);
    C=C(:,:,seq);
    LocalDensity=LocalDensity(seq);
end
ModelNumber=ModelNumber0;
[~,seq1]=ActivatingRules(ModelNumber,centerlambda,LocalDensity);
X=[1,datain];
for jj=seq1
    C(:,:,jj)=C(:,:,jj)-centerlambda(jj)*C(:,:,jj)*X'*X*C(:,:,jj)/(1+centerlambda(jj)*X*C(:,:,jj)*X');
    A1=A(:,:,jj)'+centerlambda(jj)*C(:,:,jj)*X'*(yin-X*A(:,:,jj)');
    A(:,:,jj)=A1';
end
end
function [ModelNumber,A,C,Global_mean,Global_X,center,prototype,Local_X,Support,sum_lambda,Index,Ye]=...
    SystemUpdating_st2(ModelNumber,A,C,Global_mean,Global_X,center,prototype,Local_X,Support,sum_lambda,Index,datain,yin,ii,omega,threshold,CL,W)
[centerlambda,LocalDensity]=firingstrength(datain,ModelNumber,center,Local_X,Support);
Ye=OutputGeneration(datain,A,centerlambda,ModelNumber,CL);
[~,label0]=min(pdist2(datain,center));
Support(label0)=Support(label0)+1;
center(label0,:)=((Support(label0)-1)*center(label0,:)+datain)/Support(label0);
Local_X(label0)=((Support(label0)-1)*Local_X(label0)+sum(datain.^2))/Support(label0);
[~,LocalDensity]=firingstrength(datain,ModelNumber,center,Local_X,Support);
[~,seq1]=ActivatingRules(ModelNumber,centerlambda,LocalDensity);
X=[1,datain];
for jj=seq1
    C(:,:,jj)=C(:,:,jj)-centerlambda(jj)*C(:,:,jj)*X'*X*C(:,:,jj)/(1+centerlambda(jj)*X*C(:,:,jj)*X');
    A1=A(:,:,jj)'+centerlambda(jj)*C(:,:,jj)*X'*(yin-X*A(:,:,jj)');
    A(:,:,jj)=A1';
end
end
function Ye=OutputGeneration(datain,A,centerlambda,ModelNumber,CL)
Ye=zeros(1,CL);
for ii=1:1:ModelNumber
    Ye=Ye+[1,datain]*A(:,:,ii)'*centerlambda(ii);
end
end
function [Ye]=TestingALM(data1,ModelNumber,A,center,Local_X,Support)
CL=size(A,1);
[L,W]=size(data1);
Ye=zeros(L,CL);
for ii=1:1:L
    datain=data1(ii,:);
    [centerlambda,LocalDensity]=firingstrength(datain,ModelNumber,center,Local_X,Support);
        Ye(ii,:)=OutputGeneration(datain,A,centerlambda,ModelNumber,CL);
end
end
function [centerlambda,LocalDensity]=firingstrength(datain,ModelNumber,center,Local_X,Support)
a0=Support./(Support+1);
a1=1./(Support+1);
center1=center;
Local_X1=Local_X;
LocalDensity=zeros(ModelNumber,1);
for jj=1:1:ModelNumber
    center1(jj,:)=center(jj,:)*a0(jj)+datain*a1(jj);
    Local_X1(jj)=Local_X(jj)*a0(jj)+sum(datain.^2)*a1(jj);
    AA=(Local_X1(jj)-sum(center1(jj,:).^2));
    BB=sum((datain-center1(jj,:)).^2);
    if AA==0
        LocalDensity(jj)=0;
    else
        LocalDensity(jj)=BB/AA;
    end
end
LocalDensity=exp(-1*LocalDensity);
centerlambda=LocalDensity./sum(LocalDensity);
end
function [centerlambda,seq1]=ActivatingRules(ModelNumber,centerlambda,LocalDensity)
threshold3=exp(-4);
seq1=find(LocalDensity>=threshold3)';
centerlambda(seq1)=centerlambda(seq1)./sum(centerlambda(seq1));
end