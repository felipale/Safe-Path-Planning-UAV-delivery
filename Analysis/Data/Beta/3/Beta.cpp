#include "ModelR1.h"
#include <iomanip>
#include <filesystem>
#include <iostream>
#include <unordered_map>
#include <Eigen/Dense>
#include <sstream>

float *beta_ptr;                     // ponteiro usado pela cost_function
const float alpha = 0.0035f;         // peso fixo na velocidade linear
const float dt = 0.02f;              // passo de tempo discreto (20 ms)

double cost_function(const Eigen::VectorXf& x) {
    return
        x(0)*x(0)
      + alpha * x(1)*x(1)
      + (*beta_ptr) * x(2)*x(2);
}

static std::unordered_map<int, Eigen::VectorXf> Goal;
Eigen::VectorXf goal(double current_time) {
    int idx = std::lround(current_time / dt);
    return Goal[idx];
}

bool stopping_criterion(const Eigen::VectorXf& /*x*/) {
    return false;
}

int main() {
    // ── parâmetros gerais ────────────────────────────────────────────────
    const int n_states               = 4;
    const int n_inputs               = 1;
    const int n_outputs              = 4;
    const int actions_state_possible = 7;    // discretização de ação reduzida
    const int trials                 = 3;    // uma simulação → gera apenas Data0
    const int horizon                = 15;   // horizonte de busca
    const float max_time             = 5.0f;

    // ── monta o mapa de metas (step de 0→1 em t=1s) ───────────────────────
    Eigen::VectorXf zero_vec(n_states), one_vec(n_states);
    zero_vec.setZero();
    one_vec.setZero(); one_vec(0) = 1.0f;
    for (float t = 0.0f; t <= 1.0f + 1e-6f; t += dt)
        Goal[std::lround(t / dt)] = zero_vec;
    for (float t = 1.0f + dt; t <= 40.0f + 1e-6f; t += dt)
        Goal[std::lround(t / dt)] = one_vec;

    // ── define sistema discreto ───────────────────────────────────────────
    Eigen::MatrixXf A(n_states,n_states), B(n_states,n_inputs);
    Eigen::MatrixXf C(n_outputs,n_states), D(n_outputs,n_inputs);
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

    // ── discretização das ações ───────────────────────────────────────────
    Eigen::MatrixXf discretization_actions(n_inputs,3);
    float u_min = -0.2f, u_max = +0.2f;
    float u_step = (u_max - u_min) / (actions_state_possible - 1);
    discretization_actions << u_min, u_max, u_step;

    // ── sem ruído ─────────────────────────────────────────────────────────
    Eigen::VectorXf noise(n_inputs);
    noise.setZero();

    // ── prepara o ponteiro para β ──────────────────────────────────────────
    float beta_val = 0.0f;
    beta_ptr = &beta_val;

    // ── muda para a pasta de saída ────────────────────────────────────────
    std::filesystem::current_path("C:\\Supaero\\Stage\\Codigo3\\Analysis\\Data\\Beta\\3");
    std::cout << "Dir atual agora: " << std::filesystem::current_path() << "\n";

    // ── instancia o modelo ────────────────────────────────────────────────
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

    // ── varredura de β: 0.15 < β ≤ 0.25 em passos de 0.02 ───────────────────
    //    gera arquivos Beta1.csv … Beta10.csv (valores de β = 0.15,0.17,…,0.25)
    int beta_idx = 1;
    for (beta_val = 0.20f; beta_val <= 0.30f + 1e-6f; beta_val += 0.02f, ++beta_idx) {
        std::ostringstream ss;
        ss << "Beta" << beta_idx;
        std::string base = ss.str();
        std::cout << "Rodando β = " << std::fixed << std::setprecision(2)
                  << beta_val << " … (salvando em " << base << ".csv)\n";

        // 1 simulação → gera somente base + "Data0.csv"
        model1.startSimulation(
            base,      // NameBase: Beta1, Beta2, …
            max_time,  // duração da simulação
            x0,        // estado inicial
            trials,    // 3 trial
            horizon,
            stopping_criterion
        );

        // renomeia "BetaXData0.csv" → "BetaX.csv"
        std::filesystem::rename(
            base + "Data0.csv",
            base + ".csv"
        );
    }

    return 0;
}