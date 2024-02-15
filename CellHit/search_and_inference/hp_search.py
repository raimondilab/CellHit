from ..models import AutoXGBRegressor
from ..data import DatasetLoader, GeneGetter


def prepare_data(drugID, dataset, random_state, loader, gene_selection_mode, cv_iterations=3,data_path=None):
    
    #preparer cross validation data and genes if knowledge primed
    if gene_selection_mode == 'knowledge_primed':
        
        data = []
        #obtain data for each cross validation iteration
        for i in range(cv_iterations):
            data.append(loader.split_and_scale(drugID=drugID, random_state=i))

        #get genes selected for the drug
        genes = loader.get_genes()
        gene_getter = GeneGetter(dataset=dataset, data_path=data_path, available_genes=genes)
        genes = gene_getter.get_genes(drugID)

        #subset data for the selected genes
        data_subset = []
        for idx, d in enumerate(data):
            train_X, train_y, valid_X, valid_y, test_X, test_y = d
            train_X = train_X[genes]
            valid_X = valid_X[genes]
            test_X = test_X[genes]
            data_subset.append((train_X, train_y, valid_X, valid_y, test_X, test_y))

        return data_subset, genes

    #prepare data for all genes
    elif gene_selection_mode == 'all_genes':
        #return data for all genes
        return loader.split_and_scale(drugID=drugID, random_state=0)

    #raise error if gene_selection_mode is not valid
    else:
        raise ValueError("Invalid gene_selection_mode. Must be 'knowledge_primed' or 'all_genes'")
    
def search(drugID, dataset='gdsc',
                   n_trials=300, n_startup_trials=100,
                   cv_iterations=3, num_parallel_tree=5, gpuID=0, random_state=0,
                   data_path=None, celligner_output_path=None,
                   gene_selection_mode='knowledge_primed',
                   *args, **kwargs):

    #load data
    loader = DatasetLoader(dataset=dataset,
                           data_path=data_path,
                           celligner_output_path=celligner_output_path,
                           use_external_datasets=False,
                           samp_x_tissue=2, random_state=random_state)

    # Prepare data based on gene selection mode
    if gene_selection_mode == 'knowledge_primed':
        data, genes = prepare_data(drugID, dataset, random_state, loader, gene_selection_mode, cv_iterations)
    else: 
        X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(drugID, dataset, random_state, loader, gene_selection_mode)

    # Model fitting
    xgb = AutoXGBRegressor(num_parallel_tree=num_parallel_tree, gpuID=gpuID)

    #perform hyperparameter search
    if gene_selection_mode == 'knowledge_primed':
        xgb.search(cv=data, n_trials=n_trials, n_startup_trials=n_startup_trials)
    else:
        xgb.search(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, 
                   n_trials=n_trials, n_startup_trials=n_startup_trials)

    # Select best parameters (same as before)
    best_params = max(xgb.study.best_trials, key=lambda t: t.values[1]).params
    best_params['device'] = f'cuda:{gpuID}'
    best_params['tree_method'] = 'hist'
    best_params['objective'] = 'reg:squarederror'
    best_params['eval_metric'] = 'rmse'
    best_params['num_parallel_tree'] = num_parallel_tree

    if gene_selection_mode == 'knowledge_primed':
        return best_params, genes, xgb.study
    else:
        return best_params, xgb.study

