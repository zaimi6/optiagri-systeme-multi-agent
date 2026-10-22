from pathlib import Path
import sys

from agents.coordinator_agent import CoordinatorAgent


def main() -> None:
    project_root = Path(__file__).resolve().parent
    coordinator = CoordinatorAgent(project_root=project_root)
    coordinator.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    except Exception as exc:
        print(f"Simulation failed: {exc}", file=sys.stderr)
        raise
