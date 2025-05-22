import os
import sys
import subprocess
import shutil
import datetime
import zipfile
import logging

MAIN_EXECUTABLE_NAME = "orbit_analyzer"
BATCH_EXECUTABLE_NAME = "orbit_batch_processor"
MAIN_ENTRY_SCRIPT = "controller.py"
BATCH_ENTRY_SCRIPT = "batch_processor.py"
VERSION_FILE = "version.txt"
FOLDERS_TO_CLEAN = ["build", "dist", "__pycache__"]

DOCUMENTS = [
    "docs/Orbit_Getting_Started.md",
    "docs/Orbit User Guide.md",
    "docs/Orbit Technical Guide.md",
    "docs/Orbit Deep Dive.md"
]

EXTRA_DATA = [
    ("components/schemas/component_schema.json", "components/schemas"),
    ("docs", "docs"),
    ("reports", "reports")
]

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def get_version():
    if os.path.exists(VERSION_FILE):
        with open(VERSION_FILE) as f:
            return f.read().strip()
    return datetime.datetime.now().strftime("%Y.%m.%d.%H%M")

def backup_folder(folder_name):
    if os.path.exists(folder_name):
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_name = f"{folder_name}_backup_{timestamp}"
        shutil.move(folder_name, backup_name)
        logging.info(f"Backed up '{folder_name}' to '{backup_name}'")

def check_pyinstaller():
    try:
        import PyInstaller
        logging.info("PyInstaller is already installed.")
        return True
    except ImportError:
        logging.info("PyInstaller not found. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"Error installing PyInstaller: {e}")
            return False

def build_executable(entry_script, executable_name):
    if not os.path.exists(entry_script):
        logging.error(f"Cannot find entry script: {entry_script}")
        return False

    # Create a spec file name based on the executable name
    spec_file = f"{executable_name}.spec"
    
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--name", executable_name,
        entry_script
    ]

    for src, dest in EXTRA_DATA:
        if os.path.exists(src):
            cmd.extend(["--add-data", f"{src};{dest}"])

    try:
        subprocess.check_call(cmd)
        logging.info(f"Executable '{executable_name}' built successfully.")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Build failed for {executable_name}: {e}")
        return False

def build_all_executables():
    # Build the main controller executable
    main_success = build_executable(MAIN_ENTRY_SCRIPT, MAIN_EXECUTABLE_NAME)
    
    # Build the batch processor executable
    batch_success = build_executable(BATCH_ENTRY_SCRIPT, BATCH_EXECUTABLE_NAME)
    
    return main_success and batch_success

def create_distribution_folder():
    version = get_version()
    dist_folder = f"{MAIN_EXECUTABLE_NAME}_{version}"

    if os.path.exists(dist_folder):
        backup_folder(dist_folder)
    os.makedirs(dist_folder, exist_ok=True)

    # Copy main executable
    main_exe_path = os.path.join("dist", f"{MAIN_EXECUTABLE_NAME}.exe")
    if os.path.exists(main_exe_path):
        shutil.copy2(main_exe_path, dist_folder)
        logging.info(f"Copied main executable to: {dist_folder}")
    else:
        logging.error(f"Could not find built main .exe: {main_exe_path}")
        return False

    # Copy batch processor executable
    batch_exe_path = os.path.join("dist", f"{BATCH_EXECUTABLE_NAME}.exe")
    if os.path.exists(batch_exe_path):
        shutil.copy2(batch_exe_path, dist_folder)
        logging.info(f"Copied batch processor executable to: {dist_folder}")
    else:
        logging.error(f"Could not find built batch processor .exe: {batch_exe_path}")
        return False

    # Create README with info about both executables
    with open(os.path.join(dist_folder, "README.txt"), "w") as f:
        f.write(f"{MAIN_EXECUTABLE_NAME}.exe - Run this tool to analyze individual test logs and generate reports.\n\n")
        f.write(f"{BATCH_EXECUTABLE_NAME}.exe - Run this tool to process multiple tests in batch mode.\n")
        f.write("\nExample batch usage:\n")
        f.write(f"{BATCH_EXECUTABLE_NAME}.exe --all           # Process all tests in logs directory\n")
        f.write(f"{BATCH_EXECUTABLE_NAME}.exe --tests SXM-123 SXM-456  # Process specific tests\n")
        f.write(f"{BATCH_EXECUTABLE_NAME}.exe --parallel      # Process in parallel\n")

    # Copy documentation
    for doc_path in DOCUMENTS:
        if os.path.exists(doc_path):
            shutil.copy2(doc_path, dist_folder)
            logging.info(f"Copied documentation: {doc_path}")
        else:
            logging.warning(f"Missing documentation: {doc_path}")

    # Clean up build artifacts
    for folder in FOLDERS_TO_CLEAN:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            logging.info(f"Cleaned up: {folder}")

    # Remove spec files
    for file in os.listdir():
        if file.endswith(".spec"):
            os.remove(file)
            logging.info(f"Removed spec file: {file}")

    logging.info(f"Distribution ready in: {dist_folder}")
    return True

def zip_distribution():
    version = get_version()
    dist_folder = f"{MAIN_EXECUTABLE_NAME}_{version}"
    zip_name = f"{dist_folder}.zip"

    try:
        with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(dist_folder):
                for file in files:
                    abs_path = os.path.join(root, file)
                    rel_path = os.path.relpath(abs_path, start=dist_folder)
                    zipf.write(abs_path, arcname=os.path.join(dist_folder, rel_path))
        logging.info(f"Created ZIP archive: {zip_name}")
        return True
    except Exception as e:
        logging.error(f"Failed to create ZIP: {e}")
        return False

def main():
    if not check_pyinstaller():
        return
    if not build_all_executables():
        return
    if not create_distribution_folder():
        return
    zip_distribution()
    logging.info("Build process complete.")

if __name__ == "__main__":
    main()