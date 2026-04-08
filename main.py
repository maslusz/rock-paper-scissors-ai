from importlib import import_module
import platform


PACKAGES = [
    "numpy",
    "pandas",
    "matplotlib",
    "sklearn",
    "cv2",
    "ultralytics",
    "kaggle",
]


def main() -> None:
    print("=== rock-paper-scissors-ai ===")
    print(f"Python: {platform.python_version()}")
    print("Sprawdzanie importow:")

    failed = []
    for package in PACKAGES:
        try:
            import_module(package)
            print(f"[OK] {package}")
        except Exception as exc:
            failed.append((package, exc))
            print(f"[ERR] {package}: {exc}")

    if failed:
        print("\nSrodowisko nie jest jeszcze gotowe.")
        raise SystemExit(1)

    print("\nSrodowisko wyglada OK.")


if __name__ == "__main__":
    main()
