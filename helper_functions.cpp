// Copyright 2023 Brian Howell
// MIT License
// Project: BayesOpt

#include "common.h"
#include "GaussianProcess.h"
#include "helper_functions.h"  // Include the header file

void write_to_file(bopt& b, sim& sim_set, int id, std::string file_path) {
    std::ofstream myfile;
    myfile.open(file_path + "/sim_" + std::to_string(id) + ".dat");
    myfile << "temp,rp,vp,uvi,uvt,obj,tn" << std::endl;
    myfile << b.temp << ","
           << b.rp << ","
           << b.vp << ","
           << b.uvi << ","
           << b.uvt << ","
           << b.obj << ","
           << sim_set.time_stepping
           << std::endl;
    myfile.close();
}

void store_tot_data(std::vector<bopt> &bopti,
                    sim& sim_set,
                    int num_sims,
                    std::string file_path) {
    std::cout << "--- storing data ---\n" << std::endl;
    std::ofstream myfile;
    myfile.open(file_path + "/tot_bopt.dat");
    myfile << "temp,rp,vp,uvi,uvt,obj,tn" << std::endl;
    for (int id = 0; id < num_sims; ++id) {
        myfile << bopti[id].temp << ","
               << bopti[id].rp   << ","
               << bopti[id].vp   << ","
               << bopti[id].uvi  << ","
               << bopti[id].uvt  << ","
               << bopti[id].obj  << ","
               << sim_set.time_stepping << std::endl;
    }
    myfile.close();
}

int  read_data(std::vector<bopt> &bopti,
               std::string file_path) {
    std::ifstream file(file_path + "output/tot_bopt.dat");

    std::string line;
    std::getline(file, line);  // skip first line
    int id = 0;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token;

        // create new bopt struct for data point
        bopt b;

        // Parse the comma-separated values in the line
        std::getline(iss, token, ',');
        b.temp = std::stof(token);

        std::getline(iss, token, ',');
        b.rp = std::stof(token);

        std::getline(iss, token, ',');
        b.vp = std::stof(token);

        std::getline(iss, token, ',');
        b.uvi = std::stof(token);

        std::getline(iss, token, ',');
        b.uvt = std::stof(token);

        std::getline(iss, token, ',');
        b.obj = std::stof(token);

        bopti.push_back(b);

        id++;
    }

    // return number of data points
    return id;
}

void build_dataset(std::vector<bopt> &_bopti,
                   Eigen::MatrixXd   &_x_train, Eigen::VectorXd &_y_train,
                   Eigen::MatrixXd   &_x_val,   Eigen::VectorXd &_y_val){

    // split data into training and validation sets
    int num_data = _bopti.size();
    int num_train = 0.9 * num_data;
    int num_val   = 0.1 * num_data;
    std::cout << "\n================ build dataset ================" << std::endl;
    std::cout << "num_data: " << num_data << std::endl;
    std::cout << "num_train: " << num_train << std::endl;
    std::cout << "num_val: " << num_val << std::endl;
    std::cout << "===============================================\n" << std::endl;

    // resize _x_train, _y_train, _x_val, and Y_VAL
    _x_train.resize(num_train, 5);
    _y_train.resize(num_train);
    _x_val.resize(num_val, 5);
    _y_val.resize(num_val);
    
    // shuffle dataset
    std::random_device rd;
    // std::mt19937 g(rd());
    std::mt19937 g(47);
    std::shuffle(_bopti.begin(), _bopti.end(), g);

    // initialize training and validation sets
    _x_train = Eigen::MatrixXd(num_train, 5);
    _y_train = Eigen::VectorXd(num_train);
    _x_val   = Eigen::MatrixXd(num_val, 5);
    _y_val   = Eigen::VectorXd(num_val);

    // populate training and validation sets
    for (int i = 0; i < num_train; ++i) {
        _x_train(i, 0) = _bopti[i].temp;
        _x_train(i, 1) = _bopti[i].rp;
        _x_train(i, 2) = _bopti[i].vp;
        _x_train(i, 3) = _bopti[i].uvi;
        _x_train(i, 4) = _bopti[i].uvt;
        _y_train(i)    = _bopti[i].obj;
    }

    for (int i = 0; i < num_val; ++i) {
        _x_val(i, 0) = _bopti[i + num_train].temp;
        _x_val(i, 1) = _bopti[i + num_train].rp;
        _x_val(i, 2) = _bopti[i + num_train].vp;
        _x_val(i, 3) = _bopti[i + num_train].uvi;
        _x_val(i, 4) = _bopti[i + num_train].uvt;
        _y_val(i)    = _bopti[i + num_train].obj;
    }

}


void build_dataset(std::vector<bopt> &_bopti,
                   Eigen::MatrixXd   &_x_train, 
                   Eigen::VectorXd   &_y_train){

    int num_data = _bopti.size();

    _x_train.resize(num_data, 5);
    _y_train.resize(num_data);

    // populate training and validation sets
    for (int i = 0; i < num_data; ++i) {
        _x_train(i, 0) = _bopti[i].temp;
        _x_train(i, 1) = _bopti[i].rp;
        _x_train(i, 2) = _bopti[i].vp;
        _x_train(i, 3) = _bopti[i].uvi;
        _x_train(i, 4) = _bopti[i].uvt;
        _y_train(i)    = _bopti[i].obj;
    }

}

void gen_test_points(constraints&     c, 
                     Eigen::MatrixXd& X){

    // initialize input variables
    std::random_device rd;                                          // Obtain a random seed from the hardware
    std::mt19937 gen(rd());                                         // Seed the random number generator
    std::uniform_real_distribution<double> distribution(0.0, 1.0);  // Define the range [0.0, 1.0)

    for (int ind = 0; ind<X.rows(); ++ind){
        X(ind, 0) = (c.max_temp - c.min_temp) * distribution(gen) + c.min_temp;
        X(ind, 1) = (c.max_rp - c.min_rp)     * distribution(gen) + c.min_rp;
        X(ind, 2) = (c.max_vp - c.min_vp)     * distribution(gen) + c.min_vp;
        X(ind, 3) = (c.max_uvi - c.min_uvi)   * distribution(gen) + c.min_uvi;
        X(ind, 4) = (c.max_uvt - c.min_uvt)   * distribution(gen) + c.min_uvt;
    }

}


void train_prior(GaussianProcess &MODEL, 
                 Eigen::MatrixXd &X_TRAIN, 
                 Eigen::VectorXd &Y_TRAIN,
                 std::vector<double> &M_PARAMS, 
                 int TIME_STEPPING, 
                 bool pre_learned){
    if (pre_learned){
        switch (TIME_STEPPING){
            case 0: 
                M_PARAMS = {0.99439,0.356547,0.000751229};   // obj_0 -> 673.344
                break;
            case 1: 
                M_PARAMS = {0.994256,0.623914,0.000965578};  // obj_1 -> 422.003 
                break;
            case 2:
                M_PARAMS = {0.940565,0.708302,0.000328992};  // obj_2 -> 397.977
                break;
            case 3: 
                M_PARAMS = {0.956662, 0.78564, 0.00095118};  // obj_3 -> 487.76 
                break; 
        }
        MODEL.train(X_TRAIN, Y_TRAIN, M_PARAMS);
    }
    else{
        MODEL.train(X_TRAIN, Y_TRAIN);
    }
}


void evaluate_model(GaussianProcess &MODEL, 
                    Eigen::MatrixXd &X_TEST,
                    Eigen::VectorXd &Y_TEST, 
                    bool VALIDATE){

    if (VALIDATE){
        std::cout << "--- model validation ---\n" << std::endl;
        MODEL.validate(X_TEST, Y_TEST);
        std::cout << "--- ---------------- ---\n" << std::endl;
    }
    else{
        std::cout << "--- model prediction ---\n" << std::endl;
        MODEL.predict(X_TEST, false);
        std::cout << "--- ---------------- ---\n" << std::endl;
    }
}

void sample_posterior(GaussianProcess &MODEL, 
                      Eigen::MatrixXd &X_SAMPLE, 
                      Eigen::VectorXd &Y_SAMPLE_MEAN, 
                      Eigen::VectorXd &Y_SAMPLE_STD,
                      constraints &C){

    // generate test points
    gen_test_points(C, X_SAMPLE);
    MODEL.predict(X_SAMPLE, true);
    
    Y_SAMPLE_MEAN = MODEL.get_y_test();
    Y_SAMPLE_STD  = MODEL.get_y_test_std();

}
