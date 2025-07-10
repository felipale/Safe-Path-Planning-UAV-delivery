// -----------------------------------------------------------------------------
// Este arquivo implementa uma versão simplificada de simulação de trajetória
// com horizonte de predição fixo, onde:
// 
// - `cost_function`: avalia o custo entre estado atual e um objetivo (posição, velocidade).
// - `goal`: define a referência a ser seguida (posição = 1, velocidade = 0).
// - `main`: configura o modelo dinâmico e a matriz de ações discretizadas para testes.
// 
// Esse código serve como base para testes com controle baseado em MPC ou MDPs,
// utilizando discretização de estados e cálculo de custo.
// -----------------------------------------------------------------------------

#include "Model.h"
#include <filesystem>


double cost_function(const Eigen::VectorXf& x, const Eigen::VectorXf& goal){
    
    Eigen::VectorXf relative;
    relative = goal-x;
    return relative(0)*relative(0) + 0.0035*relative(1)*relative(1); // Penaliza posição e velocidade
}

Eigen::VectorXf goal(double current_time){
    Eigen::VectorXf Goal(4);
    Goal << 1,0,0,0;                            // Objetivo fixo: posição = 1, velocidade = 0
    // return Goal[round(Ccurrent_time*100)];
    return Goal;
}

bool stopping_criterion(const Eigen::VectorXf& x, double current_time){
    // Critério de parada desabilitado (sempre retorna false)

    //    Eigen::VectorXf e;
    //    e = goal(current_time)  - x;
   return false;//e.maxCoeff() < 0.005;
}

int main(){
    const int n =4;         // Número de estados (posição e velocidade)
    const int m = 1;        // Número de entradas
    const int p = 4;        // Número de saídas
    // Matrizes do modelo dinâmico
    Eigen::MatrixXf A(n,n);
    Eigen::MatrixXf B(n,m);
    Eigen::MatrixXf C(p,n);
    Eigen::MatrixXf D(p,m);
     // Define ações possíveis: mínimo, máximo e passo de discretização
    Eigen::MatrixXf discretization_actions(m,3); //cols (Min, max, level of discretization)
    Eigen::VectorXf noise(m);                   // Vetor de ruído (aqui com valor nulo)
    
    // Sistema discreto simples com passo fixo (aproximação de integração)
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

    
     std::filesystem::current_path("C:\\Supaero\\Stage\\Codigo3\\Analysis\\Data\\Horizon\\2");
     std::cout << "Dir atual agora: "
              << std::filesystem::current_path()
              << std::endl;
              
    noise<< 0;  // Sem ruído
    int actions_state_possible = 10;
    float discrete_time_step = 0.02;
    discretization_actions << -0.2, 0.2, 0.04;  // De -0.2 a 0.2 com passo 0.04
    Eigen::VectorXf x_0(n);
    x_0 << 0,0,0,0;                                 // Estado inicial: posição 0, velocidade 0
    
    // Instancia o modelo com todas as configurações acima
    Model model1(actions_state_possible,A, B, C, D , discretization_actions, noise, cost_function, goal, discrete_time_step);
    
    int horizon =20;
    
    

    string name = "1DTeste24";
    stringstream ss;
    int pp = 0;

    // Realiza simulações com horizontes crescentes de 1 a 24
    for (horizon = 1; horizon<=24; horizon++ ){
        cout << horizon << endl;
        ss.str("");             // Limpa o conteúdo da stringstream
	    ss.clear();             // Reseta flags
        ss  <<"Horizon"<<pp;    // Gera nome como "Horizon0", "Horizon1", etc
        name = ss.str();
        
        // Inicia simulação com os parâmetros definidos
        model1.startSimulation(name, 5, x_0, 1, horizon, stopping_criterion);
        pp++;
    }

    return 0;
}