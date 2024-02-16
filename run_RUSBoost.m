%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This code is to replicate the results of RUSBoost model               %
% Our results are generated using Matlab R2020b on Windows 10           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

diary("results_rusboost.txt");
for year_test = 2003:2014
    rng(0,'twister'); % fix random seed for reproducing the results
    % read training data
    fprintf('==> Running RUSBoost (training period: %d-%d, testing period: %d, with %d-year gap)...\n',1991,year_test-2,year_test,2);
    data_train = data_reader('data_FraudDetection_JAR2020.csv','data_default',1991,year_test-2);
    y_train = data_train.labels;
    X_train = data_train.features;
    paaer_train = data_train.paaers;
    % read testing data
    data_test = data_reader('data_FraudDetection_JAR2020.csv','data_default',year_test,year_test);
    y_test = data_test.labels;
    X_test = data_test.features;
    paaer_test = unique(data_test.paaers(data_test.labels~=0));
    
    % handle serial frauds using PAAER
    y_train(ismember(paaer_train,paaer_test)) = 0;

    % train model
    t1 = tic;
    t = templateTree('MinLeafSize',5); % base model
    % see "tune_RUSBoost.m" for the tuning of the number of trees
    rusboost = fitensemble(X_train,y_train,'RUSBoost',300,t,'LearnRate',0.1,'RatioToSmallest',[1 1]);
    t_train = toc(t1);
    
    % test model
    t2 = tic;
    [label_predict,dec_values] = predict(rusboost,X_test);
    dec_values = dec_values(:,2);
    t_test = toc(t2);

    % print performance results
    fprintf('Training time: %g seconds | Testing time %g seconds \n', t_train, t_test);
    for topN = [0.01, 0.02, 0.03, 0.04, 0.05]
        metrics = evaluate(y_test,label_predict,dec_values,topN);
        fprintf('Performance (top%d%% as cut-off thresh): \n',topN*100);
        fprintf('AUC: %.4f \n', metrics.auc);
        fprintf('NCDG@k: %.4f \n', metrics.ndcg_at_k);
        fprintf('Sensitivity: %.2f%% \n', metrics.sensitivity_topk*100);
        fprintf('Precision: %.2f%% \n', metrics.precision_topk*100);
    end
end
diary off;
