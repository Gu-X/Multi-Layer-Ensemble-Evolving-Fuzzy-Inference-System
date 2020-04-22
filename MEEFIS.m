%%The MATLAB code for the research paper titled "Multi-Layer Ensemble Evolving Fuzzy Inference System".

%%This work is described in:
%===============================================================================================================================
%Gu, X. (2020). Multi-Layer Ensemble Evolving Fuzzy Inference System. IEEE Transactions on Fuzzy Systems, DOI: 10.1109/TFUZZ.2020.2988846.
%===============================================================================================================================
%%Please cite the paper above if this code helps. 
%%Programmed by Xiaowei Gu. For any queries about the code, please contact Dr. Xiaowei Gu: x.gu3@lancaster.ac.uk
function [Output]=MEEFIS(Dataset,Setting)
Layer=Setting.Layer;
Epoch=Setting.Epoch;
FF=Setting.ForgettingFactor;
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
        Input1{1,ii}.FF=FF;
        [Output1{1,ii},Systm{1,ii}]=EFIS(Input1{1,ii},Systm{1,ii},'Learning');
    end
    for jj=2:1:NumLayer
        for ii=1:1:NumElement(jj)
            Input1{jj,ii}.datain=[];
            for tt=1:1:length(Layer{jj}(ii,:))
                Input1{jj,ii}.datain=[Input1{jj,ii}.datain,Output1{jj-1,Layer{jj}(ii,tt)}.Ye];
            end
            Input1{jj,ii}.FF=FF;
            Input1{jj,ii}.dataout=Y0;
            [Output1{jj,ii},Systm{jj,ii}]=EFIS(Input1{jj,ii},Systm{jj,ii},'Learning');
        end
    end
end
Output.trainingtime=toc;
for ii=1:1:NumElement(1)
    Input2{1,ii}.datain=data1(:,Layer{1}(ii,:));
    [Output2{1,ii},Systm{1,ii}]=EFIS(Input2{1,ii},Systm{1,ii},'Testing');
end
tic
for jj=2:1:NumLayer
    for ii=1:1:NumElement(jj)
        Input2{jj,ii}.datain=[];
        for tt=1:1:length(Layer{jj}(ii,:))
            Input2{jj,ii}.datain=[Input2{jj,ii}.datain,Output2{jj-1,Layer{jj}(ii,tt)}.Ye];
        end
        [Output2{jj,ii},Systm{jj,ii}]=EFIS(Input2{jj,ii},Systm{jj,ii},'Testing');
    end
end
Output.testingtime=toc;
Output.systemparam=Systm;
Output.Ye=Output2{NumLayer,ii}.Ye;
end
