#include "ModelR1.h"
#include <iomanip>
#include <filesystem>

float *beta_ptr;                  // ponteiro que cost_function vai ler
const float alpha = 0.0035f;  // peso fixo na velocidade linear

double cost_function(const Eigen::VectorXf& x){
    
    
     
    return x(0)*x(0) + (alpha)*x(1)*x(1) + (*beta_ptr)*x(2)*x(2);
}



unordered_map<int, Eigen::VectorXf> Goal;

Eigen::VectorXf goal(double current_time){
    // return Goal[round(Ccurrent_time*100)];
    return Goal[round(current_time/0.02)];
}

bool stopping_criterion(const Eigen::VectorXf& x){
//    Eigen::VectorXf e;
//    e = goal(current_time)  - x;
   return false;//e.maxCoeff() < 0.005;
}

int main(){

    float time;
     Eigen::VectorXf axx(4);
     for(time = 0; time<=1; time+=0.02){
          // cout << time << endl;
          axx<< 0,0,0,0;
          Goal[round(time/0.02)] = axx;
     }
     for(time = 1+0.02 ; time<=40; time+=0.02){
          // cout << time << endl;
          axx << 1,0,0,0;
          Goal[round(time/0.02)] = axx;
     }

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

D    << 0,
        0,
        0,
        0;


     std::filesystem::current_path("C:\\Supaero\\Stage\\Codigo3\\Analysis\\Data\\Beta\\1");
     std::cout << "Dir atual agora: "
              << std::filesystem::current_path()
              << std::endl;    

    noise<< 0;
    int actions_state_possible = 10;
    float discrete_time_step = 0.02;
    discretization_actions << -0.2, 0.2, 0.04;
    Eigen::VectorXf x_0(n);
    x_0 << 0,0,0,0;
    Model model1(actions_state_possible, A, B, C, D, discretization_actions, noise, cost_function, goal, 0.02);
    
    
    int horizon =25;
    int trials = 5;
    float a = 0;
    float beta_val = 0.0f;
    beta_ptr = &beta_val;

    string name = "1DTeste24";
    stringstream ss;
    int pp = 0;

    for (beta_val = 0.0f; beta_val<=0.5f; beta_val+=0.02f ){
        cout << a << endl;
          std::ostringstream ss;
          ss << "Beta" 
               << std::fixed << std::setprecision(2) 
               << beta_val;
          std::cout << "Rodando β=" << beta_val << " …\n";      
          model1.startSimulation(
               ss.str(),            // NameBase
               5.0f,                // max_time
               x_0,                 // estado inicial
               trials,              // número de simulações
               horizon,             // horizonte
               stopping_criterion   // critério de parada
          );
    }

    return 0;
}