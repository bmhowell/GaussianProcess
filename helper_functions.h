// Copyright 2023 Brian Howell
// MIT License
// Project: BayesOpt

#ifndef SRC_HELPER_FUNCTIONS_H_
#define SRC_HELPER_FUNCTIONS_H_
#include <vector>
#include <string>
#include "common.h"
#include "GaussianProcess.h"


// declare functions
int     find_arg_idx(int argc, char** argv, const char* option);

void    write_to_file(bopt &b,
                      sim& sim_set,
                      int id,
                      std::string file_path);

void    store_tot_data(std::vector<bopt> &bopti,
                       sim &sim_set,
                       int num_sims,
                       std::string file_path);

int     read_data(std::vector<bopt> &bopti,
                  std::string file_path); 

void    build_dataset(std::vector<bopt> &_bopti,
                      Eigen::MatrixXd   &_x_train, Eigen::VectorXd &_y_train,
                      Eigen::MatrixXd   &_x_val,   Eigen::VectorXd &_y_val); 


void    build_dataset(std::vector<bopt> &_bopti,
                      Eigen::MatrixXd   &_x_train, 
                      Eigen::VectorXd   &_y_train); 


void    gen_test_points(constraints &c, Eigen::MatrixXd &X); 

void    genetic_algorithm(std::vector<double> &OPT_C, 
                          std::vector<double> &INIT_VALS, 
                          FunctionPtr         OBJ_FUNC,
                          std::string         FILE_PATH);


void    train_prior(GaussianProcess &MODEL, 
                    Eigen::MatrixXd &X_TRAIN, 
                    Eigen::VectorXd &Y_TRAIN,
                    std::vector<double> &M_PARAMS, 
                    int TIME_STEPPING, 
                    bool pre_learned);

void    evaluate_model(GaussianProcess &MODEL, 
                       Eigen::MatrixXd &X_TEST, 
                       Eigen::VectorXd &Y_TEST, 
                       bool VALIDATE);

void    sample_posterior(GaussianProcess &MODEL, 
                         Eigen::MatrixXd &X_SAMPLE, 
                         Eigen::VectorXd &Y_SAMPLE_MEAN, 
                         Eigen::VectorXd &Y_SAMPLE_STD,
                         constraints &C);

#endif  // SRC_HELPER_FUNCTIONS_H_
