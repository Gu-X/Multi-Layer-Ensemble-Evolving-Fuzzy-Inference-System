clear all
clc
close all

load ExampleDataset_regression % load the example

%% Define the interconnections between different layers (this is only an example used in the paper).
L=length(X_training(1,:));
seq=[1:1:L,1:1:L];
for tt=1:1:4
    ii=(tt-1)*floor(L/4)+1;
    Layer{1}(tt,:)=seq(ii:1:ii+ceil(L/2)-1);
end
seq2=[1:1:length(Layer{1}(:,1)),1:1:length(Layer{1}(:,1))];
for ii=1:1:4
    Layer{2}(ii,:)=seq2(ii:1:ii+length(Layer{1}(:,1))-2);
end
Layer{3}=[1:1:length(Layer{1}(:,1))];
%% Define parameters.
Setting.Layer=Layer; % Layers.
Setting.Epoch=1; % Number of training epochs.
Setting.ForgettingFactor=0.1; % Forgettingfactor.
Dataset.training=X_training; % Training data.
Dataset.traininglabels=Y_training;% Ground truth.
Dataset.testing=X_testing; % Testing data.
[Output]=MEEFIS(Dataset,Setting);  % Get the estimated output

rmse=sqrt(mean((Output.Ye-Y_testing).^2))
