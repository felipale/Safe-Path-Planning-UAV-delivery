#include "ModelR1.h"
#include <iomanip>
#include <filesystem>
#include <iostream>
#include <unordered_map>
#include <Eigen/Dense>
#include <sstream>

// ── ponteiros para os pesos ──────────────────────────────────────────────

float *delta_ptr;                    // δ, peso na velocidade angular
const float alpha = 0.000f;         // α, peso na velocidade linear
const float dt    = 0.02f;           // passo de tempo discreto

// ── função de custo agora com 4 termos ─────────────────────────────────
double cost_function(const Eigen::VectorXf& x) {
    // x = [ erro_y; erro_ẏ; erro_β; erro_β̇ ]
    return
        x(0)*x(0)                      // (y−y_ref)^2, peso=1
      + (*delta_ptr) * x(3)*x(3);     // δ(β̇−β̇_ref)^2
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
    const int actions_state_possible = 5;  // reduzido de 7 → 5
    const int trials                 = 5; // reduzido de 5 → 3
    const int horizon                = 5;  // reduzido de 15 → 5

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
    A <<  1.0f, dt,     -3.27345e-06f, 5.00671e-05f,
          0.0f, 1.0f,   -0.000491158f, 0.00500924f,
          0.0f, 0.0f,    0.998035f,     0.020037f,
          0.0f, 0.0f,   -0.196563f,     1.00304f;
    B <<  0.00196527f,
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

    // ── fixa β e inicializa δ ─────────────────────────────────────────────
    float delta_val = 0.0f;           // δ inicia em 0
    delta_ptr = &delta_val;

    // ── muda para a pasta de saída ────────────────────────────────────────
    std::filesystem::current_path(
      "C:\\Supaero\\Stage\\Codigo3\\Analysis\\Data\\DeltaC\\5"
    );
    std::cout << "Dir atual agora: "
              << std::filesystem::current_path() << "\n";

    // ── instancia o modelo ────────────────────────────────────────────────
    Eigen::VectorXf x0(n_states); x0.setZero();
    Model model1(
        actions_state_possible,
        A, B, C, D,
        discretization_actions,
        noise,
        cost_function,
        goal,
        dt
    );

    // ── varredura de δ: 0.0 ≤ δ ≤ -1.0 em passos de 0.1 ───────────────────
    int delta_idx = 1;
    for (delta_val = 0.0f;
         delta_val >= -1.0f - 1e-6f;
         delta_val -= 0.1f, ++delta_idx)
    {
        std::ostringstream ss;
        ss << "Delta" << delta_idx;
        std::string base = ss.str();

        std::cout << "Rodando δ = "
                  << std::fixed << std::setprecision(2)
                  << delta_val
                  << "\n";

        model1.startSimulation(
            base,       // NameBase: Delta1, Delta2, …
            5.0f,       // duração da simulação
            x0,         // estado inicial
            trials,     // numero de trials
            horizon,
            stopping_criterion
        );
    }

    return 0;
}
