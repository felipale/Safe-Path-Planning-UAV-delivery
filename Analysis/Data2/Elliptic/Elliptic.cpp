#include "ModelR1.h"
#include <iomanip>
#include <filesystem>
#include <iostream>
#include <unordered_map>
#include <Eigen/Dense>
#include <vector>
#include <sstream>

// ── coeficientes globais ───────────────────────────────────────────────
float alpha_val = 0.0035f;  // α = 0.0035
float beta_val  = 0.00001f;     // β
float delta_val = 0.000065f;     // δ

double cost_function(const Eigen::VectorXf& x) {
    return
        x(0)*x(0)
      + alpha_val * x(1)*x(1)
      + beta_val  * x(2)*x(2)
      + delta_val * x(3)*x(3);
}

static std::unordered_map<int, Eigen::VectorXf> Goal;
Eigen::VectorXf goal(double current_time) {
    int idx = std::lround(current_time / 0.02);
    return Goal[idx];
}

bool stopping_criterion(const Eigen::VectorXf& /*x*/) {
    return false;
}

int main() {
    // ── parâmetros ───────────────────────────────────────────────────────
    const float dt                     = 0.02f;
    const int   n_states               = 4;
    const int   n_inputs               = 1;
    const int   n_outputs              = 4;
    const int   actions_state_possible = 4;
    const int   trials                 = 7;
    const int   horizon                = 5;
    const float max_time               = 20.0f;   // duração da trajetória elíptica

    // ── define e entra na pasta de saída ─────────────────────────────────
    const auto base_output =
        std::filesystem::path("C:\\Supaero\\Stage\\Codigo3\\Analysis\\Data2\\Elliptic");
    std::filesystem::create_directories(base_output);
    std::filesystem::current_path(base_output);
    std::cout << "Dir atual agora: " << std::filesystem::current_path() << "\n";

    // ── inicializa o mapa de referência com trajetória elíptica ──────────
    {
        Eigen::VectorXf axx(n_states);
        const float f = 1.0f;  // razão dos eixos da elipse
        for (float t = 0.0f; t <= max_time + dt + 1e-6f; t += dt) {
            axx << std::cos(t),    // x = cos(t)
                   -f * std::sin(t), // ẋ = -f·sin(t)
                    std::sin(t),    // y = sin(t)
                    f * std::cos(t);  // ẏ =  f·cos(t)
            Goal[std::lround(t/dt)] = axx;
        }
    }

    // ── modelo linear discreto ───────────────────────────────────────────
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

    // ── discretização das ações (min, max, step) ───────────────────────
    Eigen::MatrixXf discretization_actions(n_inputs,3);
    float u_min = -0.2f, u_max = +0.2f;
    float u_step = (u_max - u_min) / (actions_state_possible - 1);
    discretization_actions << u_min, u_max, u_step;

    // ── ruído na entrada (1 canal) ──────────────────────────────────────
    Eigen::VectorXf noise(n_inputs);
    noise << 0.03f;

    // ── instancia o modelo ──────────────────────────────────────────────
    Model model(
        actions_state_possible,
        A, B, C, D,
        discretization_actions,
        noise,
        cost_function,
        goal,
        dt
    );

    // ── estado inicial (zero) ───────────────────────────────────────────
    Eigen::VectorXf x0(n_states);
    x0.setZero();

    // ── executa simulação única com trajetória elíptica ────────────────
    const std::string run_label = "Elliptic2D";
    std::cout << "Salvando em: " << base_output
              << "  (prefix: " << run_label << ")\n";

    model.startSimulation(
        run_label,
        max_time,
        x0,
        trials,
        horizon,
        stopping_criterion
    );

    return 0;
}
