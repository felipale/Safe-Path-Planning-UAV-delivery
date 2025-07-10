#include "Model.h"

double cost_function(const Eigen::VectorXf& x, const Eigen::VectorXf& goal){
    
    Eigen::VectorXf relative;
    relative = goal-x;
    return relative(0)*relative(0) + 0.0035*relative(1)*relative(1);
}

Eigen::VectorXf goal(double current_time){
    Eigen::VectorXf Goal(4);
    Goal << 1,0,0,0;
    // return Goal[round(Ccurrent_time*100)];
    return Goal;
}

bool stopping_criterion(const Eigen::VectorXf& x, double current_time){
//    Eigen::VectorXf e;
//    e = goal(current_time)  - x;
   return false;//e.maxCoeff() < 0.005;
}

       

int main(){
    const int n = 4;
    const int m = 1;
    const int p = 4;
    Eigen::MatrixXf A(n,n);
    Eigen::MatrixXf B(n,m);
    Eigen::MatrixXf C(p,n);
    Eigen::MatrixXf D(p,m);
    Eigen::MatrixXf discretization_actions(m,3); //cols (Min, max, level of discretization)
    Eigen::VectorXf noise(m);
    
    
A    <<    1,         0.02, -3.27345e-06,  5.00671e-05,
           0,            1, -0.000491158,   0.00500924,
           0,            0,     0.998035,     0.020037,
           0,            0,    -0.196563,      1.00304; 
B    << 0.00196527,
        0.196691,
        0.00196463,
        0.196563;

C    << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1;

D    <<  0,
        0,
        0,
        0;

    noise<< 0.023;
    int actions_state_possible = 10;
    float discrete_time_step = 0.02;
    discretization_actions << -0.2, 0.2, 0.04;
    Eigen::VectorXf x_0(n);
    x_0 << 0, 0, 0, 0;
    
    Model model1(actions_state_possible,A, B, C, D , discretization_actions, noise, cost_function, goal, discrete_time_step);
    
    int horizon = 10;
    model1.startSimulation("Noise003", 5, x_0, 100, horizon, stopping_criterion);
    

    return 0;
}