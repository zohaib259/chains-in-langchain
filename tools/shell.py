import subprocess

try:
    result = subprocess.run(
        ["powershell", "-Command", "ls"], 
        capture_output=True, 
        text=True, 
        check=True
    )
    print(result.stdout)
except subprocess.CalledProcessError as e:
    print(f"Error executing command: {e.stderr}")