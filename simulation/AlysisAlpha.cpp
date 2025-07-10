#include "Model.h"

// -----------------------------------------------------------------------------
// Este arquivo realiza um estudo paramétrico sobre o impacto do peso α na
// função de custo quadrático usada para o controle de trajetória.
//
// - A função de custo penaliza erro de posição e de velocidade, com α = 0.02.
// - Isso permite avaliar como diferentes pesos afetam o comportamento do sistema.
// - O restante do código é semelhante a outras simulações com horizonte fixo.
//
// Pode ser usado para gerar gráficos de sensibilidade ou estabilidade do controle.
// -----------------------------------------------------------------------------

// Função de custo quadrático com peso α = 0.02 para a velocidade
double cost_function(const Eigen::VectorXf& x, const Eigen::VectorXf& goal){
    
    Eigen::VectorXf relative;
    relative = goal-x;
    return relative(0)*relative(0) + 0.02*relative(1)*relative(1);
}

// unordered_map<int, Eigen::VectorXf> Goal; 


// Função que retorna o estado desejado (goal)
// Aqui fixo: posição = 1, velocidade = 0
Eigen::VectorXf goal(double current_time){
    Eigen::VectorXf Goal(2);
    Goal << 1,0;
    // return Goal[round(Ccurrent_time*100)];
    return Goal;
}

// Critério de parada — desativado

bool stopping_criterion(const Eigen::VectorXf& x, double current_time){
//    Eigen::VectorXf e;
//    e = goal(current_time)  - x;
   return false;//e.maxCoeff() < 0.005;
}


int main(){
    const int n =2;
    const int m = 1;
    const int p = 2;
    Eigen::MatrixXf A(n,n);
    Eigen::MatrixXf B(n,m);
    Eigen::MatrixXf C(p,n);
    Eigen::MatrixXf D(p,m);
    Eigen::MatrixXf discretization_actions(m,3); //cols (Min, max, level of discretization)
    Eigen::VectorXf noise(m);
    
    
    A << 1, 0.2,
         0, 1;
    

    B << 0.001962,
         0.1962;


    C << 1, 0,
         0, 1; 
    
    D << 0,
         0;

    

    noise<< 0.03;
    int actions_state_possible = 10;              // número de ações discretas
    float discrete_time_step = 0.02;              // passo de tempo da simulação
    discretization_actions << -0.2, 0.2, 0.04;    // discretização de ações
    Eigen::VectorXf x_0(n);
    x_0 << 0,0;                                   // estado inicial
    
    // Instancia o modelo com as configurações acima
    Model model1(actions_state_possible,A, B, C, D , discretization_actions, noise, cost_function, goal, discrete_time_step);
    cout << "Hello from teste" << endl;
    model1.update_state_discretization(0.0001);
    int horizon =20;

     // Executa a simulação para o horizonte fixo e escreve saída com nome fixo
    //model1.update_state_discretization(0.000001);
    model1.startSimulation("1D", 5, x_0, 1, horizon, stopping_criterion);

    return 0;
}