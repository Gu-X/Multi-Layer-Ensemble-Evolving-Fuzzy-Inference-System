clear all
clc
close all
clear function

load ExampleDataset_classification % load the example
%% Define the interconnections between different layers (this is only an example used in the paper).
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
%% Define parameters.
Setting.Layer=Layer; % Layers.
Setting.Epoch=1; % Number of training epochs.
Setting.ForgettingFactor=0; % Forgettingfactor.
Dataset.training=X_training; % Training data.
Dataset.traininglabels=full(ind2vec(Y_training')'); % Ground truth.
Dataset.testing=X_testing; % Testing data.
[Output]=MEEFIS(Dataset,Setting); %% Run meefis
[~,Output]=max(Output.Ye,[],2); % Get the estimated output
confusionmat(Y_testing,Output) % confusion matrix
sum(sum(confusionmat(Y_testing,Output).*eye(length(unique(Y_testing)))))/length(Y_testing) % classification accuracy
