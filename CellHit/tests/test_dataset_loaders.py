import pytest
import sys
import numpy as np
sys.path.append('..')  # Adjust the path as necessary  # Adjust the path as necessary
from data import DatasetLoader  # Adjust the import statement based on your project structure
import importlib.util



#new Dataset Object
@pytest.fixture
def dataset_loader():
    data_path = '../../data/'
    celligner_output_path = '../../data/transcriptomics/celligner_CCLE_TCGA.feather'

    loader = DatasetLoader(dataset='gdsc',
                            data_path=data_path,
                            celligner_output_path=celligner_output_path,
                            use_external_datasets=False,
                            samp_x_tissue=2,
                            random_state=0)

    return loader


#old Dataset Object
@pytest.fixture
def old_dataset_loader():
    data_path = '/home/fcarli/CellSense/data/'
    project_path = '/home/fcarli/CellSense/'
    module_path = "/home/fcarli/CellSense/data.py"
    class_name = "DataGenerator"

    # Load the module from the specified path
    spec = importlib.util.spec_from_file_location("data", module_path)
    data_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data_module)

    # Access the OldLoader class from the loaded module
    DataGenerator = getattr(data_module, class_name)

    data_generator = DataGenerator(dataset='gdsc',data_path=data_path,project_path=project_path,
                                avail_genes=None,drug_threshold=10,
                                samp_x_tissue=2,random_state=0)
    
    return data_generator

#test that 
def test_dataset_disjoint_splits(dataset_loader):

    # Activate split_and_scale with val_split=True to generate train, validation, and test splits
    dataset_loader.split_and_scale(val_split=True, val_random_state=0)

    # Extract DepMapID sets
    train_depmap_ids = set(dataset_loader.meta_train['DepMapID'])
    val_depmap_ids = set(dataset_loader.meta_valid['DepMapID'])
    test_depmap_ids = set(dataset_loader.meta_test['DepMapID'])

    # Assertions
    assert train_depmap_ids.isdisjoint(val_depmap_ids), "Train and Validation sets share DepMapID(s)"
    assert train_depmap_ids.isdisjoint(test_depmap_ids), "Train and Test sets share DepMapID(s)"
    assert val_depmap_ids.isdisjoint(test_depmap_ids), "Validation and Test sets share DepMapID(s)"


#test backward compatibility with old code
def test_single_drug(dataset_loader,old_dataset_loader):

    train_X,train_Y,test_X,test_Y,valid_X,valid_Y = dataset_loader.split_and_scale(drugID=1909,val_split=True, val_random_state=0)
    old_train_X,old_train_Y,old_valid_X,old_valid_Y,old_test_X,old_test_Y = old_dataset_loader.get_splits(drug_id=1909)

    #get meta_train,meta_valid,meta_test dataframes from each object and compare them
    train_X.sort_index(inplace=True), old_train_X.sort_index(inplace=True)
    test_X.sort_index(inplace=True), old_test_X.sort_index(inplace=True)
    valid_X.sort_index(inplace=True), old_valid_X.sort_index(inplace=True)
    
    #assert that the index of the dataframes are the same
    assert train_X.index.equals(old_train_X.index), "Train X dataframes have different indices"
    assert test_X.index.equals(old_test_X.index), "Test X dataframes have different indices"
    assert valid_X.index.equals(old_valid_X.index), "Valid X dataframes have different indices"
    assert train_X.columns.equals(old_train_X.columns), "Train X dataframes have different columns"
    assert test_X.columns.equals(old_test_X.columns), "Test X dataframes have different columns"
    assert valid_X.columns.equals(old_valid_X.columns), "Valid X dataframes have different columns"

    assert np.allclose(train_X,old_train_X,rtol=1e-03, atol=1e-08), "Train X dataframes are different"
    assert np.allclose(test_X,old_test_X,rtol=1e-03, atol=1e-08), "Test X dataframes are different"
    assert np.allclose(valid_X,old_valid_X,rtol=1e-03, atol=1e-08), "Valid X dataframes are different"
    assert np.allclose(train_Y.values,old_train_Y,rtol=1e-03, atol=1e-08), "Train Y dataframes are different"
    assert np.allclose(test_Y.values,old_test_Y,rtol=1e-03, atol=1e-08), "Test Y dataframes are different"
    assert np.allclose(valid_Y.values,old_valid_Y,rtol=1e-03, atol=1e-08), "Valid Y dataframes are different"