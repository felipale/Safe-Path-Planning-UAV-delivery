#include "ModelR1.h"
#include <filesystem>

double cost_function(const Eigen::VectorXf& x){
    double a =x(0)*x(0) + 0.0035*x(1)*x(1);
    a += x(4)*x(4) + 0.0035*x(5)*x(5);
 
    return a;
//     return x(0)*x(0) + 0.012*x(1)*x(1);
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
     
     // força o diretório de saída para C:\Supaero\Stage\...
     std::filesystem::current_path("C:\\Supaero\\Stage\\Codigo3\\Analysis\\DataFinal\\ElipticCoupled");
     std::cout << "Salvando resultados em: " 
               << std::filesystem::current_path() 
               << std::endl;
     // inicialize Goal
     float f = 1;
     float time;
     Eigen::VectorXf axx(8);
     for(time = 0 ; time<=40; time+=0.02){
          // cout << time << endl;
          axx << cos(time),-f*sin(time),0,0, sin(time), f*cos(time),0,0;
          Goal[round(time/0.02)] = axx;
     }

     




    const int n = 8;
    const int m = 2;
    const int p = 8;
    Eigen::MatrixXf A(n,n);
    
    Eigen::MatrixXf B(n,m);
    Eigen::MatrixXf C(p,n);
    Eigen::MatrixXf D(p,m);
    Eigen::MatrixXf discretization_actions(m,3); //cols (Min, max, level of discretization)
    Eigen::VectorXf noise(m);
    
    
    A << 
        1.0f,       0.02f,    -3.27345e-06f,  5.00671e-05f,  0.0f,    0.0f,    0.0f,     0.0f,
        0.0f,       1.0f,     -4.91158e-04f,  5.00924e-03f,  0.0f,    0.0f,    0.0f,     0.0f,
        0.0f,       0.0f,      0.998035f,      0.020037f,    0.0f,    0.0f,    0.0f,     0.0f,
        0.0f,       0.0f,     -0.196563f,      1.00304f,     0.0f,    0.0f,    0.0f,     0.0f,
        0.0f,       0.0f,      0.0f,           0.0f,         1.0f,    0.02f,  -3.27345e-06f, 5.00671e-05f,
        0.0f,       0.0f,      0.0f,           0.0f,         0.0f,    1.0f,  -4.91158e-04f, 5.00924e-03f,
        0.0f,       0.0f,      0.0f,           0.0f,         0.0f,    0.0f,   0.998035f,    0.020037f,
        0.0f,       0.0f,      0.0f,           0.0f,         0.0f,    0.0f,  -0.196563f,    1.00304f;

    B << 
       -2.04214e-05f,  0.0f,
       -2.04384e-03f,  0.0f,
       -2.04147e-05f,  0.0f,
       -2.04250e-03f,  0.0f,
        0.0f,         2.04214e-05f,
        0.0f,         2.04384e-03f,
        0.0f,         2.04147e-05f,
        0.0f,         2.04250e-03f;

    C = Eigen::MatrixXf::Identity(p, n);
    D = Eigen::MatrixXf::Zero(p, m);

    
    noise<< 0.03, 0.03;
    int actions_state_possible = 4;
    float discrete_time_step = 0.02;
    cout << "Teste " << endl;
    discretization_actions << -0.2, 0.2, 0.04,
                              -0.2, 0.2, 0.04;
                              
    Eigen::VectorXf x_0(n);
    x_0 << 0,0,0,0,0,0,0,0;
    
    Model model1(actions_state_possible,A, B, C, D , discretization_actions, noise, cost_function, goal, discrete_time_step);
    
    int horizon = 2;
    model1.startSimulation("2DelpticH22", 20, x_0, 5, horizon, stopping_criterion);
//     for (auto& g: Goal){
//           cout << g.first*0.02 << "- "<< g.second.transpose() << endl;
//     }

    return 0;
}
