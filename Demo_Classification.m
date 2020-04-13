clear all
% clc
close all
clear function

load ExampleDataset_classification

L=length(X_training(1,:));
seq=[1:1:L,1:1:L];
count=1;
for tt=1:1:4
    ii=(tt-1)*floor(L/4)+1;
    Layer{1}(count,:)=seq(ii:1:ii+ceil(L/2)-1);
    count=count+1;
end
seq2=[1:1:length(Layer{1}(:,1)),1:1:length(Layer{1}(:,1))];
for ii=1:1:4
    Layer{2}(ii,:)=seq2(ii:1:ii+length(Layer{1}(:,1))-2);
end
Layer{3}=[1:1:length(Layer{1}(:,1))];
%%
Setting.Layer=Layer;
Setting.Epoch=1;
Setting.FogettingFactor=0;
Dataset.training=X_training;
Dataset.traininglabels=full(ind2vec(Y_training')');
Dataset.testing=X_testing;
[Output]=MEEFIS(Dataset,Setting);
[~,Output]=max(Output.Ye,[],2);
confusionmat(Y_testing,Output)
sum(sum(confusionmat(Y_testing,Output).*eye(length(unique(Y_testing)))))/length(Y_testing)