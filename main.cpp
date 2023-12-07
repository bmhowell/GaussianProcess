// Copyright 2023 Brian Howell
// MIT License
// Project: GP

#include "GaussianProcess.h"
#include "common.h"
#include "helper_functions.h"

int main(int argc, char** argv) {
    //////////////  PARSE ARGS  //////////////
    // opt constraints (default) and sim settings (default)
    constraints c;
    sim         s;
    s.bootstrap = false;
    s.time_stepping = 4;
    s.update_time_stepping_values();

    // MACBOOK PRO
    std::string file_path;
    file_path = "/Users/brianhowell/Desktop/Berkeley/MSOL/gaussian_process/";

    // data storage
    std::vector<bopt> bopti;

    // STEP 1: retrieve data set
    int ndata0;
    bool multi_thread = true;
    if (!s.bootstrap) {
        ndata0 = read_data(bopti, file_path);
    }
    std::cout << "Number of data points: " << ndata0 << std::endl;

    // STEP 2: initialize function approximator and optimizer
    int n_dim = 5; 
    // convert data to Eigen matrices
    Eigen::MatrixXd x_train;
    Eigen::VectorXd y_train;
    Eigen::VectorXd y_train_std;

    Eigen::MatrixXd x_test; 
    Eigen::VectorXd y_test;
    Eigen::VectorXd y_test_std;

    build_dataset(bopti, x_train, y_train, x_test, y_test);

    // specify model {RBF, RQK, LOC_PER}
    GaussianProcess model = GaussianProcess("LOC_PER", file_path);

    // STEP 3: train model
    model.train(x_train, y_train);
    
    // STEP 4: validate
    const bool validate = true; 
    evaluate_model(model, x_test, y_test, validate);

    // STEP 5: predict
    const bool compute_std = true;
    model.predict(x_test, compute_std);
    std::cout << "Predicted values: \n" << model.get_y_test().transpose().head(5) << std::endl;
    std::cout << "Predicted uncertainty: \n" << model.get_y_test_std().transpose().head(5) << std::endl;
}

// Command Line Option Processing
int find_arg_idx(int argc, char** argv, const char* option) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], option) == 0) {
            return i;
        }
    }
    return -1;
}
