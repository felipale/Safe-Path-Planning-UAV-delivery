
//--------------------------------------------------------------------
//Esse arquivo é pra beta acoplado, ou seja, vamos considerar alpha =0
//E analisar a influência de beta sobre o sistema
//--------------------------------------------------------------------

#include "ModelR1.h"
#include <iomanip>
#include <filesystem>
#include <iostream>
#include <unordered_map>
#include <Eigen/Dense>
#include <sstream>   // para std::ostringstream

float *beta_ptr;                     // ponteiro usado pela cost_function
const float alpha = 0.00f;           // peso fixo na velocidade linear = 0
const float dt = 0.02f;              // passo de tempo discreto (20 ms)

double cost_function(const Eigen::VectorXf& x) {
    // custo quadrático nos estados, ponderado por alpha e beta
    return
        x(0)*x(0)
        + (*beta_ptr) * x(2)*x(2);
}

static std::unordered_map<int, Eigen::VectorXf> Goal;
Eigen::VectorXf goal(double current_time) {
    // converte tempo contínuo em índice de 20 ms
    int idx = std::lround(current_time / dt);
    return Goal[idx];
}

bool stopping_criterion(const Eigen::VectorXf& /*x*/) {
    return false;  // sem critério de parada customizado
}

int main() {
    // --- parâmetros de discretização e simulação ---
    const int n_states               = 4;
    const int n_inputs               = 1;
    const int n_outputs              = 4;
    const int actions_state_possible = 7;  // reduzido de 10 → 7
    const int trials                 = 7;   // reduzido de 10 → 7
    const int horizon                = 5;  // reduzido de 15 → 5

    // gera objetivos nos primeiros 1 s (zero) e depois até 40 s (um)
    Eigen::VectorXf zero_vec(n_states), one_vec(n_states);
    zero_vec.setZero();
    one_vec.setZero(); one_vec(0) = 1.0f;

    for (float t = 0.0f; t <= 1.0f; t += dt) {
        Goal[std::lround(t / dt)] = zero_vec;
    }
    for (float t = 1.0f + dt; t <= 40.0f; t += dt) {
        Goal[std::lround(t / dt)] = one_vec;
    }

    // --- definição do sistema discreto ---
    Eigen::MatrixXf A(n_states, n_states), B(n_states, n_inputs);
    Eigen::MatrixXf C(n_outputs, n_states), D(n_outputs, n_inputs);
    A <<
         1.0f,    dt,      -3.27345e-06f,  5.00671e-05f,
         0.0f,    1.0f,    -0.000491158f,  0.00500924f,
         0.0f,    0.0f,     0.998035f,      0.020037f,
         0.0f,    0.0f,    -0.196563f,      1.00304f;
    B <<
         0.00196527f,
         0.196691f,
         0.00196463f,
         0.196563f;
    C.setIdentity();
    D.setZero();

    // --- discretização da ação de controle ---
    // min, max, e passo calculado para 7 níveis uniformes em [-0.2, +0.2]
    Eigen::MatrixXf discretization_actions(n_inputs, 3);
    float u_min  = -0.2f, u_max = +0.2f;
    float u_step = (u_max - u_min) / (actions_state_possible - 1);
    discretization_actions << u_min, u_max, u_step;

    // sem ruído
    Eigen::VectorXf noise(n_inputs);
    noise.setZero();

    // define beta inicial e atribui ponteiro
    float beta_val = 0.00f;
    beta_ptr = &beta_val;

    // caminho de saída (ajuste conforme sua estrutura de pastas)
    std::filesystem::current_path("C:\\Supaero\\Stage\\Codigo3\\Analysis\\Data\\BetaC\\3");
    std::cout << "Dir atual agora: " << std::filesystem::current_path() << "\n";

    // inicialização do modelo
    Eigen::VectorXf x0(n_states);
    x0.setZero();
    Model model1(
        actions_state_possible,
        A, B, C, D,
        discretization_actions,
        noise,
        cost_function,
        goal,
        dt
    );

    // varredura macro: beta de 0.000 → 0.01 em passos de 0.001,
    // mas nomeando arquivos como Beta1, Beta2, ...
    int beta_idx = 1;
    for (beta_val = 0.00f; beta_val <= 0.01f + 1e-6f; beta_val += 0.001f, ++beta_idx) {
        std::ostringstream ss;
        ss << "Beta" << beta_idx;
        std::cout << "Rodando β = " << std::fixed << std::setprecision(1)
                  << beta_val << " … (salvando em " << ss.str() << ".csv)\n";

        model1.startSimulation(
            ss.str(),      // NameBase simples: Beta1, Beta2, ...
            5.0f,          // max_time (s)
            x0,            // estado inicial
            trials,        // número de simulações
            horizon,       // horizonte de busca
            stopping_criterion
        );
    }

    return 0;
}
