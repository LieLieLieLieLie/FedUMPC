"""Run the partial-client-participation stress test."""

from paper_experiments import Config, _gen_data, reset_rng, run_participation_stress


if __name__ == "__main__":
    Config.set_mode("fast")
    reset_rng(100)
    client_data = _gen_data()
    run_participation_stress(client_data)
