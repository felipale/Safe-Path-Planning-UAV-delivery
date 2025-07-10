#include "ModelR1.h"
#include <filesystem>

double cost_function(const Eigen::VectorXf& x){
    return x(0)*x(0) + 0.0035*x(1)*x(1);
}

unordered_map<int, Eigen::VectorXf> Goal;

Eigen::VectorXf goal(double current_time){
    return Goal[round(current_time/0.02)];
}

bool stopping_criterion(const Eigen::VectorXf& x){
    return false;
}

int main(){
     // força o diretório de saída para C:\Supaero\Stage\...
    std::filesystem::current_path("C:\\Supaero\\Stage\\Codigo3\\Analysis\\DataFinal\\Cosine");
    std::cout << "Salvando resultados em: " 
            << std::filesystem::current_path() 
            << std::endl;
    // inicialize Goal
    Eigen::VectorXf axx(4);
    for(int k=0; k<=110/0.02; ++k){
        double time = k*0.02;
        axx << cos(time), -sin(time), cos(time), -sin(time);
        Goal[k] = axx;
    }

    const int n = 4;
    const int m = 1;
    const int p = 4;
    Eigen::MatrixXf A(n,n);
    Eigen::MatrixXf B(n,m);
    Eigen::MatrixXf C(p,n);
    Eigen::MatrixXf D(p,m);
    Eigen::MatrixXf discretization_actions(m,3);
    Eigen::VectorXf noise(m);

    // Novo modelo 4×4
    A <<  1,         0.02f,       -3.27345e-06f,  5.00671e-05f,
          0,            1.0f,      -0.000491158f,   0.00500924f,
          0,            0,         0.998035f,       0.020037f,
          0,            0,        -0.196563f,       1.00304f;

    B << 0.00196527f,
         0.196691f,
         0.00196463f,
         0.196563f;

    C << 1,0,0,0,
         0,1,0,0,
         0,0,1,0,
         0,0,0,1;

    D << 0,
         0,
         0,
         0;

    noise << 0.03f;
    int actions_state_possible = 4;
    float discrete_time_step = 0.02f;
    discretization_actions << -0.2f, 0.2f, 0.04f;

    Eigen::VectorXf x_0(n);
    x_0 << 0,0,0,0;

    Model model1(actions_state_possible,
                 A, B, C, D,
                 discretization_actions,
                 noise,
                 cost_function,
                 goal,
                 discrete_time_step);

    int horizon = 10;
    model1.startSimulation("Cos1H4", 60, x_0, 5, horizon, stopping_criterion);

    return 0;
}
