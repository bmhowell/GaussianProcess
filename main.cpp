// Copyright 2023 Brian Howell
// MIT License
// Project: GP

#include "GaussianProcess.h"
#include "common.h"
#include "helper_functions.h"

int main(int argc, char** argv) {
    auto start = std::chrono::high_resolution_clock::now();

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
    GaussianProcess model = GaussianProcess("RBF", file_path);

    // STEP 3: train model
    std::vector<double> model_param = {0.983407, 0.804468, 0.000923096};
    const bool pre_learned = true;
    train_prior(model,
                x_train, 
                y_train, 
                model_param, 
                s.time_stepping, 
                pre_learned);
    
    // STEP 4: validate
    const bool validate = true; 
    evaluate_model(model, x_test, y_test, validate);

    // STEP 5: predict
    const bool compute_std = true;
    model.predict(x_test, compute_std);
    std::cout << "Predicted values: \n" << model.get_y_test().transpose().head(5) << std::endl;
    std::cout << "Predicted uncertainty: \n" << model.get_y_test_std().transpose().head(5) << std::endl;

    // Get the current time after the code segment finishes
    auto end = std::chrono::high_resolution_clock::now();
    auto t = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    auto duration = t.count() / 1e6;
    std::cout << "\n---Time taken by code segment: "
              << duration  / 60
              << " min---" << std::endl;


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
