// main.cpp

#include "ModelR1.h"
#include <iomanip>
#include <filesystem>
#include <iostream>
#include <unordered_map>
#include <Eigen/Dense>
#include <sstream>

// ── ponteiros para os pesos ──────────────────────────────────────────────
float *alpha_ptr;                    // α
float *beta_ptr;                     // β
float *delta_ptr;                    // δ
const float dt = 0.02f;              // passo de tempo discreto

// ── função de custo com 4 termos ─────────────────────────────────────────
double cost_function(const Eigen::VectorXf& x) {
    return
        x(0)*x(0)
      + (*alpha_ptr) * x(1)*x(1)
      + (*beta_ptr)  * x(2)*x(2)
      + (*delta_ptr) * x(3)*x(3);
}

// ── mapa de metas temporais ──────────────────────────────────────────────
static std::unordered_map<int, Eigen::VectorXf> Goal;
Eigen::VectorXf goal(double current_time) {
    int idx = std::lround(current_time / dt);
    return Goal[idx];
}

bool stopping_criterion(const Eigen::VectorXf& /*x*/) {
    return false;
}

// formata em mili (prefixo + valor*1000 + "m")
std::string format_mili(const std::string& prefix, float val) {
    int mili = static_cast<int>(val * 1000.0f + 0.5f); // arredondamento
    return prefix + std::to_string(mili) + "m";
}

// formata em micro (prefixo + valor*1e6 + "u")
std::string format_micro(const std::string& prefix, float val) {
    int micro = static_cast<int>(val * 1e6f + 0.5f);
    return prefix + std::to_string(micro) + "u";
}

int main() {
    // ── parâmetros gerais ────────────────────────────────────────────────
    const int    n_states               = 4;
    const int    n_inputs               = 1;
    const int    n_outputs              = 4;
    const int    actions_state_possible = 6;
    const int    trials                 = 3;
    const int    horizon                = 5;
    const float  max_time               = 5.0f;

    // ── metas (step de 0→1 em t=1s) ───────────────────────────────────────
    Eigen::VectorXf zero_vec(n_states), one_vec(n_states);
    zero_vec.setZero();
    one_vec.setZero(); one_vec(0) = 1.0f;
    for (float t = 0.0f; t <= 1.0f + 1e-6f; t += dt)
        Goal[std::lround(t / dt)] = zero_vec;
    for (float t = 1.0f + dt; t <= 40.0f + 1e-6f; t += dt)
        Goal[std::lround(t / dt)] = one_vec;

    // ── sistema discreto ─────────────────────────────────────────────────
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

    // ── pesos e ponteiros ─────────────────────────────────────────────────
    float alpha_val = 0.0f;
    float beta_val  = 0.0f;
    float delta_val = 0.0f;
    alpha_ptr = &alpha_val;
    beta_ptr  = &beta_val;
    delta_ptr = &delta_val;

    // ── pasta de saída ────────────────────────────────────────────────────
    std::filesystem::current_path("C:\\Supaero\\Stage\\Codigo3\\Analysis\\Data\\Combination");
    std::cout << "Dir atual agora: " << std::filesystem::current_path() << "\n";

    Eigen::VectorXf x0(n_states); x0.setZero();

        // ── laço de varredura ─────────────────────────────────────────────────
        int sim_id = 0;
        for (alpha_val = 0.002f; alpha_val <= 0.004f + 1e-6f; alpha_val += 0.001f) {
            for (beta_val = 0.0000f; beta_val <= 0.0002f + 1e-9f; beta_val += 0.00005f) {
                for (delta_val = 0.005f; delta_val <= 0.02f + 1e-6f; delta_val += 0.005f) {

                    std::string alpha_str = format_mili("A", alpha_val);
                    std::string beta_str  = format_micro("B", beta_val);
                    std::string delta_str = format_mili("D", delta_val);
                    std::string prefix    = alpha_str + "_" + beta_str + "_" + delta_str;

                    std::cout << "[Sim " << sim_id++ << "] " << prefix << "\n";

                    Model model1(
                        actions_state_possible,
                        A, B, C, D,
                        discretization_actions,
                        noise,
                        cost_function,
                        goal,
                        dt
                    );

                    // chamo todas as trials de uma vez, sem sobrescrita
                    model1.startSimulation(
                        prefix,       // só NameBase
                        max_time,
                        x0,
                        trials,       // gera Data0, Data1 e Data2
                        horizon,
                        stopping_criterion
                    );
                }
            }
        }

    return 0;
}
